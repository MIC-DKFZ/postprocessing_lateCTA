import os,sys
import pandas as pd
import numpy as np
import argparse
import time
from scipy.stats import rankdata
from batchgenerators.utilities.file_and_folder_operations import save_json
from joblib import Parallel, delayed

def process_id(df : pd.DataFrame) -> np.ndarray:
    """
    Obtain rank of scores for each ID

    Params
    ------
    df : series to be ranked

    Returns
    -------
    rank : resulting rank of Dice scores
    
    """
    rank = rankdata(-df.values, method='min')
    return rank


def main(args):
    infolder = args.i
    outfile = args.o
    workers = args.np

    assert os.path.exists(infolder), f"Input folder with Dice dataframes '{infolder}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)), f"Parent folder of output file '{os.path.dirname(outfile)}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Iterate through dataframes and concatenate them
    files = sorted(os.listdir(infolder))
    dfs = [pd.read_csv(os.path.join(infolder, file)) for file in files if ".csv" in file]
    df = pd.concat(dfs)
    df.set_index('Unnamed: 0', inplace=True)

    # Iterate through IDs to obtain rankings
    ids = df.index.astype(str).tolist()
    cols = np.array(list(df.columns), dtype=str)
    rank_matrix = np.array(Parallel(n_jobs=workers)(delayed(process_id)(df.loc[i]) for i in ids))

    # Aggregate ranks
    rank_mean = np.mean(rank_matrix, axis=0)
    argmin = np.argmin(rank_mean) # Best working column

    # Set up comparison between dist=0.0 column and optimal column found
    best_col = cols[argmin]
    df_out = df[["dist_0.0", best_col]]

    cols_out = list(df_out.columns)
    for col_out in cols_out:
        print(df_out[col_out].mean())

    # Save params
    params = {}
    best_col_split = best_col.split("_")
    if "dist" in best_col:
        ind_dist = best_col_split.index('dist')
        best_dist = best_col_split[ind_dist+1]
        params['dist'] = best_dist
    if 'logit' in best_col:
        ind_logit = best_col_split.index('logit')
        best_logit = best_col_split[ind_logit+1]
        params['logit'] = best_logit
        
    save_json(params, outfile)

    df_out.to_csv(outfile.replace('.json', '.csv'))

def get_args():
    # Set up best cutoff logit and distance threshold
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input folder with Dice dataframes", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    parser.add_argument("--np", help="Workers", default=6, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
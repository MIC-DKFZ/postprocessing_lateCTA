import os,sys
import numpy as np
import argparse
import pandas as pd
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import write_data



def main(args):
    # Set patients labelled manually in the validation set 
    qa_file = args.qa
    d_folder = args.d
    outfile = args.out

    assert os.path.exists(qa_file), f"QA file '{qa_file}' does not exist"
    assert os.path.exists(d_folder), f"Discarded IDs folder '{d_folder}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)), f"Parent directory of output file '{os.path.dirname(outfile)}' does not exist"

    # Load QA file and derive manual IDs
    qa_df = pd.read_csv(qa_file, header=None)
    labelling = qa_df.iloc[:,-2].astype(str) 
    ind_manual = np.where(labelling == "manual")[0]
    cids = qa_df.iloc[:,0].astype(str)

    # Load discarded IDs
    discarded_files = sorted(os.listdir(d_folder))
    cids_discarded = [f.replace(".json", "") for f in discarded_files if ".json" in f]
    cids = np.setdiff1d(cids, np.array(cids_discarded, dtype=str))  
    cids_manual = cids[ind_manual]

    cids_automated = np.setdiff1d(cids, cids_manual)

    # Set up output dictionary with train and validation sets
    out = [{"train" : cids_automated.tolist(), 
            "val" : cids_manual.tolist()}]
    # Write json file  
    write_data(data = out, filename = outfile)
    # Write pkl file 
    write_data(data = out, filename = outfile.replace(".json", ".pkl"))
    

def get_args():
    # Prepare validation set for nnDet experiment as .json and .pkl file
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", help="File with QA data", required=True, type=str)
    parser.add_argument("--d", help="Folder with discarded IDs after QA", required=True, type=str)
    parser.add_argument("--out", help="Output .json file", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(time.time()-t1)
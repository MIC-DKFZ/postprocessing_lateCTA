import pandas as pd
import os,sys
import numpy as np
import argparse
# import matplotlib.pyplot as plt
import shutil

def extract_defective_ids(qa_df : pd.DataFrame) -> list:
    """
    Obtain defective IDs to be removed from
    the subsequent analysis

    Params
    ------
    qa_df : QA information

    Returns
    -------
    defective_ids : set of IDs to be removed
    
    """
    ids = qa_df.iloc[:,0].values.astype(str) 
    minimum_corr = qa_df.iloc[:,-3].values
    ind = np.where(minimum_corr < 0.85)[0] 
    defective_ids = ids[ind]

    return defective_ids 


def main(args):
    qa = args.qa
    out = args.out

    assert os.path.exists(qa), f"QA file '{qa}' does not exist"
    assert os.path.exists(os.path.dirname(out)), f"Parent folder of '{out}' does not exist"
    assert ".txt" in out, f"Output file '{out}' is not TXT"

    # Read QA file with pandas
    qa_df = pd.read_csv(qa, delimiter=",", header=None)

    # Obtain defective IDs
    defective_ids = extract_defective_ids(qa_df=qa_df)

    
    for i in defective_ids:
        in_folder = "/scratch/amartinezmora/raw_data/ctp_time"
        out_folder = "/scratch/amartinezmora/raw_data/temp"
        file = os.path.join(in_folder, f"{i}_AcquisitionDateTime.npy")
        outfile = os.path.join(out_folder, f"{i}_AcquisitionDateTime.npy")
        if os.path.exists(file):
            shutil.move(file, outfile)
        else:
            file = os.path.join(in_folder, f"{i}_AcquisitionTime.npy")
            if os.path.exists(file):
                outfile = os.path.join(out_folder, f"{i}_AcquisitionTime.npy")
                shutil.move(file, outfile)
    

    # Save resulting file
  #  np.savetxt(out, defective_ids, fmt="%s")    


def get_args():
    # Identify those case IDs from the registration process to be removed for further preprocessing 
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", help="File with QA information from registration process", required=True, type=str)
    parser.add_argument("--out", help="Output file with IDs", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
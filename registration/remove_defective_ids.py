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
    reincluded = args.reincluded
    base_dir = args.base_dir

    assert os.path.exists(qa), f"QA file '{qa}' does not exist"
    assert os.path.exists(os.path.dirname(out)), f"Parent folder of '{out}' does not exist"
    assert ".txt" in out, f"Output file '{out}' is not TXT"
    assert os.path.exists(base_dir), f"Base directory '{base_dir}' does not exist"

    # Manage reincluded IDs, if filename has been provided 
    reincluded_ids = None
    if reincluded is not None:
        assert ".txt" in reincluded, f"Reincluded file '{reincluded}' is not TXT"
        assert os.path.exists(reincluded), f"Reincluded file '{reincluded}' does not exist"
        reincluded_ids = np.loadtxt(reincluded, dtype=str)

    # Read QA file with pandas
    qa_df = pd.read_csv(qa, delimiter=",", header=None)

    # Obtain defective IDs
    defective_ids = extract_defective_ids(qa_df=qa_df)

    # Read reintroduced IDs, if file is available 
    if reincluded_ids is not None:
        defective_ids = np.setdiff1d(defective_ids, reincluded_ids)

    # Move CTA, CTP and time information from defective IDs into specific folders
    for i in defective_ids:
        # Handle data folders 
        cta_folder_in = os.path.join(base_dir, "cta")
        cta_folder_out = os.path.join(base_dir, "cta_discarded")
        ctp_folder_in = os.path.join(base_dir, "ctp_registered_to_cta")
        ctp_folder_out = os.path.join(base_dir, "ctp_registered_to_cta_discarded")
        time_folder_in = os.path.join(base_dir, "ctp_time")
        time_folder_out = os.path.join(base_dir, "ctp_time_discarded")

        if not(os.path.exists(cta_folder_out)):
            os.makedirs(cta_folder_out)
        
        if not(os.path.exists(ctp_folder_out)):
            os.makedirs(ctp_folder_out)

        if not(os.path.exists(time_folder_out)):
            os.makedirs(time_folder_out)

        cta_file_in = os.path.join(cta_folder_in, f"{i}.nii.gz")
        cta_file_out = os.path.join(cta_folder_out, f"{i}.nii.gz")
        ctp_subfolder_in = os.path.join(ctp_folder_in, i)
        ctp_subfolder_out = os.path.join(ctp_folder_out, i)
        time_file_in = os.path.join(time_folder_in, f"{i}_AcquisitionDateTime.npy")
        time_file_out = os.path.join(time_folder_out, f"{i}_AcquisitionDateTime.npy")

        if os.path.exists(cta_file_in):
            # Move CTA file 
            shutil.move(cta_file_in, cta_file_out)

        if os.path.exists(time_file_in):
            # Move time file
            shutil.move(time_file_in, time_file_out)
        else:
            time_file_in = os.path.join(time_folder_in, f"{i}_AcquisitionTime.npy") 
            time_file_out = os.path.join(time_folder_out, f"{i}_AcquisitionTime.npy") 
            if os.path.exists(time_file_in):
                 shutil.move(time_file_in, time_file_out)

        if os.path.exists(ctp_subfolder_in):
            # Move CTP subfolder 
            shutil.move(ctp_subfolder_in, ctp_subfolder_out) 


def get_args():
    # Identify those case IDs from the registration process to be removed for further preprocessing 
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", help="File with QA information from registration process", required=True, type=str)
    parser.add_argument("--out", help="Output file with IDs", required=True, type=str)
    parser.add_argument("--reincluded", help="File with reincluded IDs that were originally discarded, built by hand", type=str, default=None)
    parser.add_argument("--base_dir", help="Basis directory where to find CTA, ctp, and time data", type=str, required=True)
   
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
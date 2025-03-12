import os,sys
from registration.setParameters import set_parameters
import argparse
from loguru import logger
from utils.load_save import load_data
import pandas as pd
import shutil
from register import execute_registration


def obtain_reference_frames(qa_file : os.PathLike, ctp_folder : os.PathLike, ctp_ref_folder : os.PathLike):
    """
    Obtain files with the CTP frames containing the largest correlation 
    to the reference CTA data, obtained from QA register script

    Params
    ------
    qa_file : file with QA information of the CTP-CTA registration
    ctp_folder : folder with registered CTP data
    ctp_ref_folder : folder where to store reference frame data
    
    """
    # Read QA file
    qa_df = pd.read_csv(qa_file, header=None, delimiter=",")

    # Iterate through case IDs
    ids = qa_df.iloc[:,0].values.astype(str)
    frame_max = qa_df.iloc[:,-2].values 
    for i,f in zip(ids, frame_max):
        outfile = os.path.join(ctp_ref_folder, f"{i}.nii.gz")
        if not(os.path.exists(outfile)):
            str_f = "00" + str(f)
            src_file = os.path.join(ctp_folder,i,f"{i}_t_{str_f[(-2):]}.nii.gz")
            shutil.copyfile(src_file, outfile)




def main(args):
    # Load arguments
    ctp_folder = args.ctp
    ctp_ref_folder = args.ctp_ref
    qa_file = args.qa
    out_folder = args.out

    assert os.path.exists(ctp_folder), f"Folder with CTP data '{ctp_folder}' does not exist"
    assert os.path.exists(qa_file), f"File with QA information '{qa_file}' does not exist"
    assert os.path.exists(os.path.dirname(ctp_ref_folder)), f"Parent folder of CTP reference folder '{ctp_ref_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent folder of output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    if not(os.path.exists(ctp_ref_folder)):
        os.makedirs(ctp_ref_folder)

    # Set up logfile
    logfile = os.path.join(os.path.dirname(out_folder),"register.log")
    logger.add(logfile, level="INFO")

    # Load config
    config_file = "register_config.json"
    assert os.path.exists(config_file), f"Configuration file for registration '{config_file}' does not exist"
    cfg = load_data(filename=config_file)
    cfg_keys = list(cfg.keys())

    # Set up registration parameters
    assert "method" in cfg_keys, "Key 'method' absent in configuration"
    assert "default_val" in cfg_keys, "Key 'default_val' absent in configuration"
    assert "metric" in cfg_keys, "Key 'metric' absent in configuration"
    
    p = set_parameters(method=cfg["method"], 
                       DefaultPixelValue=cfg["default_val"], 
                       metric=cfg["metric"])
    
    # Retrieve reference frames from the CTP  
    obtain_reference_frames(qa_file=qa_file, ctp_folder=ctp_folder, ctp_ref_folder=ctp_ref_folder)

    # Iterate through cases in reference CTP folder
    ctp_ref_files = sorted(os.listdir(ctp_ref_folder))
    for ctp_ref_file in ctp_ref_files:
        if ".nii.gz" in ctp_ref_file:
            full_ctp_ref_file = os.path.join(ctp_ref_folder, ctp_ref_file)
            execute_registration(cta_file=full_ctp_ref_file, 
                                ctp_folder=ctp_folder, 
                                out_folder=out_folder, 
                                cfg=cfg, 
                                p=p)
            
            


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctp", help="Folder with already registered CTP data", required=True, type=str)
    parser.add_argument("--ctp_ref", help="Folder where to store reference CTP frame data", required=True, type=str)
    parser.add_argument("--qa", help="File with QA information from CTP-to-CTA registration step", required=True, type=str)
    parser.add_argument("--out", help="Output folder with corrected data", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())

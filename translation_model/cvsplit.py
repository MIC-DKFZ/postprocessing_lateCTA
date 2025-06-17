import os, sys
import pandas as pd
import argparse
from loguru import logger
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import write_data, load_data

def phase_loading(file : os.PathLike, cids : list):
    """
    Load phase information
    
    Params
    ------
    file : phase file
    cids : case IDs of interest

    Returns
    -------
    phase_info : dict
    
    """
    # Load phase info
    assert os.path.exists(file), f"Phase file '{file}' does not exist"
    phase = load_data(filename=file)

    # Locate case IDs of interest in phase file
    phase_cids = np.array(list(phase.keys()), 
                          dtype=str)
    cids = np.intersect1d(cids, 
                          phase_cids)
    
    # Provide final data
    phase_info = {cid : phase[cid] for cid in cids}

    return phase_info


def cv_split(args):
   # Apply cross-val split

   # Argument loading
   task_folder = args.task
   strat = bool(args.strat)

   # Derive configuration file
   cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "cfg_translation.json")

   # Preprocessed folder
   preprocessed_folder = os.path.join(os.getenv("data_folder"),
                                          "preprocessed",
                                          task_folder,
                                          "imagesTr")
   
   assert os.path.exists(preprocessed_folder), f"Preprocessed folder '{preprocessed_folder}' does not exist" 

   assert (
       os.path.exists(cfg_file) and ".json" in cfg_file
   ), f"Configuration file '{cfg_file}' does not exist or is not .json"
   cfg = load_data(cfg_file)
   cfg_keys = list(cfg.keys())

   assert "cv" in cfg_keys, "Key 'cv' not in configuration file"
   folds = cfg["cv"]

   # Set up logger
   logger.remove()
   logger.add(
       sys.stdout,
       format="<level>{level}</level>: {message}",
       level="INFO",
       colorize=True,
   )
   log_file = os.path.join(os.path.dirname(preprocessed_folder), "cv_split.log")
   logger.add(log_file, level="INFO")
   

   # Load case IDs in preprocessed folder
   prep_files = sorted(os.listdir(preprocessed_folder))
   cids = np.array([prep_file.replace("_0000.b2nd","") for prep_file in prep_files if "_0000.b2nd" in prep_file], 
                   dtype=str)

   # Load phase information, in case of stratified fold splitting
   if strat:
       phase_file = os.path.join(os.getenv("data_folder"),
                                 "phase.json")
       phase_info = phase_loading(file=phase_file,
                                  cids=cids)
       
       # Create Stratified K-Folds
       skf = StratifiedKFold(n_splits=folds, 
                             shuffle=True, 
                             random_state=42)

       # Store split information
       splits = [{"train" : cids[train_idx].tolist(), "val" : cids[val_idx].tolist()} for train_idx,val_idx in skf.split(cids, phase_info)]

   else:
       # No stratification applied
       kf = KFold(n_splits=folds, 
                  shuffle=True,
                  random_state=42)

       # Store split information
       splits = [{"train" : cids[train_idx].tolist(), "val" : cids[val_idx].tolist()} for train_idx,val_idx in kf.split(cids)]


   # Set up output file
   outfile = os.path.join(os.path.dirname(preprocessed_folder), 
                          "splits_final.json")
   write_data(data=splits, 
              filename=outfile)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task folder", type=str)
    parser.add_argument("--strat", help="Stratification", type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    cv_split(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
    
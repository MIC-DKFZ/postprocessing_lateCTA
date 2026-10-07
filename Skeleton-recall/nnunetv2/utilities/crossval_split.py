from typing import List

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import os,sys
from batchgenerators.utilities.file_and_folder_operations import load_json

def strat_loading(file : os.PathLike, cids : list):
    """
    Load information for stratification
    
    Params
    ------
    file : phase file
    cids : case IDs of interest

    Returns
    -------
    info : dict
    
    """
    # Load phase info
    assert os.path.exists(file), f"Phase file '{file}' does not exist"
    phase = load_json(filename=file)

    # Locate case IDs of interest in phase file
    phase_cids = np.array(list(phase.keys()), 
                          dtype=str)
    cids = np.intersect1d(cids, 
                          phase_cids)
    
    # Provide final data
    info = {cid : phase[cid] for cid in cids}

    return info


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits


def generate_stratified_crossval_split(train_identifiers: List[str], file_strat : os.PathLike, seed=12345, n_splits=5) -> List:
    
    # Load stratification information as dict
    strat = strat_loading(file=file_strat, 
                          cids=train_identifiers)
    
    # We assume the stratification file contains information for all train identifiers 
    
    # Create Stratified K-Folds
    skf = StratifiedKFold(n_splits=n_splits, 
                            shuffle=True, 
                            random_state=42)

    # Store split information
    train_identifiers = np.array(train_identifiers,
                                 dtype=str)
    splits = [{"train" : train_identifiers[train_idx].tolist(), "val" : train_identifiers[val_idx].tolist()} for train_idx,val_idx in skf.split(list(strat.keys()), list(strat.values()))]
    

    return splits

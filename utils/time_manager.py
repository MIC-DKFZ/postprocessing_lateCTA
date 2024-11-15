import numpy as np
import os,sys
from typing import Union


def load_time(time_folder : os.PathLike, cid : str) -> np.ndarray:
    """
    Load time array information

    Params
    ------
    time_folder : folder with time files
    cid : case ID to be extracted

    Returns
    -------
    t : time information for case of interest
    
    """
    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    # Obtain time files
    time_files = np.array(sorted(os.listdir(time_folder)), dtype=str)

    cid_files = time_files[np.char.find(time_files, cid) >= 0]
    assert cid_files.shape[0] == 1, f"Either none or more than one corresponding case ID files were found in time folder '{time_folder}'"  

    # Load time information from found file
    time_file = os.path.join(time_folder, cid_files[0])
    time_info = np.load(time_file)

    # Take time information from mid-slice and set the offset to zero
    t = time_info[time_info.shape[0] // 2]
    t -= t.min()
    return t



def extract_ids(folder : os.PathLike) -> list:
    """
    Extract case IDs from CTP folder

    Params
    ------
    folder : folder with CTP information

    Returns
    -------
    ids : case IDs
    
    """
    ids = sorted(os.listdir(folder))
    ids = [i for i in ids if os.path.isdir(os.path.join(folder,i))]
    return ids


def extract_time_resolution(folder : os.PathLike, time_folder : os.PathLike) -> Union[dict, float]:
    """
    Extract time resolution for all CTP scans

    Params
    ------
    folder : folder with CTP files
    time_folder : folder with time files

    Returns
    -------
    times : dict with all time vectors for all cases
    delta_t : median time resolution found
    
    """
    # Derive case IDs
    cids = extract_ids(folder= folder)

    # Load all time files and resolutions
    times = {}
    resolutions = []
    for cid in cids:
        times[cid] = load_time(time_folder=time_folder, cid=cid)
        resolution = np.abs(np.diff(times[cid]))
        resolutions += [resolution]

    # Define target resolution as median of all time resolutions found
    delta_t = np.median(np.array(resolutions))
    
    return times, delta_t
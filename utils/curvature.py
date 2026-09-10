# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os,sys
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt


def extract_inflection_points(x: np.ndarray, y : np.ndarray) -> np.ndarray:
    """
    Get points with maximum curvature, before the curve maximum is reached

    Params
    ------
    x : time points to be processed
    y : curve to be processed (AIF)

    Returns
    -------
    t_inflect : computed inflection points through time

    """
    # Derive the time when the maximum value of Y is reached, since the inflection points 
    # should occur only before the maximum

    dy = np.gradient(y, x)        # First derivative
    d2y = np.gradient(dy, x)      # Second derivative

    # Inflection points are where the second derivative changes sign
    curvature = np.diff(np.sign(d2y))
    inflection_points = np.where(curvature)[0]  

    # filter inflection points before the curve's main maximum
    argmax = np.argmax(y)
    if argmax == 0:
        # Recompute in case that the argmax happens just at the beginning
        argmax = np.argmax(y[1:]) + 1
          
    inflection_points = inflection_points[inflection_points < argmax]

    # Consider the inflection point with the biggest second derivative
    d2y_inflection = np.abs(d2y[inflection_points])
    ind_inflection = np.argmax(d2y_inflection)

    final_ind = inflection_points[ind_inflection]

    # Consider a couple of slices earlier than the inflection point
    # In case the AIF mask is being taken a bit later in the circulation
    if final_ind > 2:
        final_ind -= 2

    t_inflect = x[final_ind]

    return t_inflect


def second_cycle(x : np.ndarray, y : np.ndarray, thr_prominence : float = 20):
    """
    Determine if there is a second cycle in the AIF function

    Params
    ------
    x : time points to be processed
    y : curve to be processed (AIF)
    thr_prominence : threshold to consider a second peak found as prominent

    Returns
    -------
    t_prominent : time moment when a second peak happens
    
    """
    t_prominent = None # default values

    # Determine argmax
    argmax = np.argmax(y)

    # Determine peaks
    peaks = find_peaks(x = y, height=None)[0]

    if peaks.shape[0] > 0:
        # Consider only peaks after the argmax 
        peaks = peaks[peaks > argmax]
    
        if peaks.shape[0] > 0:
            # Determine prominences
            y_peaks = y[peaks] 
            prominence = peak_prominences(x = y, peaks=peaks)[0]

            # Determine relative prominences 
            prominence_rel = prominence*100/(y_peaks + np.finfo(float).eps)

            # Determine if some point exceeds the 
            # threshold of relative prominence 
            # If so, take the peak with the highest relative prominence
            inds = np.where(prominence_rel > thr_prominence)[0]
            if inds.shape[0] > 0:
                # Estimate prominence time 
                prominence_rel_argmax = np.argmax(prominence_rel)
                t_prominent = x[peaks[prominence_rel_argmax]] 
                

    return t_prominent


def aifs2cutoffs(aif_folder : os.PathLike, time_folder : os.PathLike, thr_prominence : float = 20.) -> dict:
    """
    Obtain cutoff times and secondary peak times from AIF folder

    Params
    ------
    aif_folder : folder with AIF curve information
    time_folder : folder with time file information
    thr_prominence : threshold with %prominence information 
        to find secondary peaks in the AIF curves (default: 20)
    
    """
    from time_manager import load_time
    assert os.path.exists(aif_folder), f"AIF folder '{aif_folder}' does not exist"
    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"

    # Load files
    files = sorted(os.listdir(aif_folder))

    # Define output dictionary
    info = {}  
    for file in files:
        # Load ID from each file 
        if ".npy" in file:
            cid = file.replace(".npy", "")
            full_file = os.path.join(aif_folder, file)
            # Load AIF and time information 
            aif = np.load(full_file)
            times = load_time(time_folder = time_folder, 
                              cid=cid)

            # Derive cutoff and secondary peaks
            cutoff = extract_inflection_points(x = times, y = aif)
            secondary = second_cycle(x = times, y = aif, 
                                     thr_prominence=thr_prominence) 
            if secondary is None:
                secondary = "nan"

            info[cid] = {"cutoff" : cutoff, 
                         "secondary" : secondary}
            
    return info

    
            

if __name__ == "__main__":
    from load_save import write_data, load_data
    folder = "/scratch/amartinezmora/preprocessed/aif"
    folder_time = "/scratch/amartinezmora/raw_data/ctp_time"
    outfile = "/scratch/amartinezmora/preprocessed/aif/cutoff_secondary_info.json"
    cfg_file = "/scratch/amartinezmora/code/config.json"

    assert os.path.exists(os.path.dirname(outfile)), f"Parent folder of output file '{outfile}' does not exist"
    assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist"

    # Load configuration
    cfg = load_data(filename = cfg_file)
    assert "thr_prominence" in list(cfg.keys()), "key 'thr_prominence' not found in configuration"
    thr_prominence = cfg["thr_prominence"] 

    # Load cutoff and secondary peak information
    info = aifs2cutoffs(aif_folder=folder, 
                        time_folder=folder_time,
                        thr_prominence=thr_prominence) 
    write_data(data = info, filename=outfile) 

    """
    file_time = "t.npy"
    file_aif = "aif.npy"
    t = np.load(file_time)
    aif = np.load(file_aif)

    t_inflect = extract_inflection_points(x=t, y=aif)
    t_prominent = second_cycle(x = t, y = aif)

    plt.figure()
    plt.plot(t,aif)
    plt.axvline(x = t_inflect, color="r")
    plt.axvline(x = t_prominent, color="r")
    plt.show()
    """
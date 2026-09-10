# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os,sys
from typing import Union
from scipy.interpolate import interp1d
from loguru import logger


def load_time(time_folder : os.PathLike, cid : str) -> Union[np.ndarray, np.ndarray]:
    """
    Load time array information

    Params
    ------
    time_folder : folder with time files
    cid : case ID to be extracted

    Returns
    -------
    t : time information for case of interest
    time_info : time information matrix
    
    """
    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    # Obtain time files
    time_files = np.array(sorted(os.listdir(time_folder)), dtype=str)

    cid_files = time_files[np.char.find(time_files, cid) >= 0]
    if cid_files.shape[0] == 0:
        logger.info(f"No time file found for case ID '{cid}', skipping...")
        return None
    
    #assert cid_files.shape[0] == 1, f"Either none or more than one corresponding case ID files were found in time folder '{time_folder}'"  

    # Load time information from found file
    time_file = os.path.join(time_folder, cid_files[0])
    time_info = np.load(time_file)

    # Take time information from mid-slice and set the offset to zero
    t = time_info[time_info.shape[0] // 2]
    t -= t.min()

    return t, time_info



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
    ids = ["_".join(i.split("_")[:(-1)]) for i in ids]
    return ids


def extract_time_resolution(folder : os.PathLike, time_folder : os.PathLike, default_delta : float = 1.5) -> Union[dict, float]:
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
    cids = extract_ids(folder= time_folder)

    # Load all time files and resolutions
    times = {}
    resolutions = []
    for cid in cids:
        times_cid, time_info = load_time(time_folder=time_folder, cid=cid)
        if times_cid is not None:
            # times[cid] = times_cid
            times[cid] = time_info 
            # resolution = np.abs(np.diff(times[cid]))
            diff = np.abs(np.diff(time_info, 1))
            resolution = np.median(diff, 1)
            resolutions += resolution.tolist()


    # Define target resolution as median of all time resolutions found
    delta_t = default_delta # Default resolution if no time information is found
    if len(resolutions) > 0: 
        delta_t = np.median(np.array(resolutions))
    
    return times, delta_t


def resample_time(ctp_array : np.ndarray, times : np.ndarray, delta_t : float, interp_type : str = "linear") -> Union[np.ndarray, np.ndarray]:
    """
    Resample input CTP array to a certain time resolution
    given the time indexes of the given case

    Params
    ------
    
    ctp_array : input CTP
    times : time points of input CTP
    delta_t : time resolution
    output : folder where to store array with resampled times
    cid : case ID of interest being analyzed
    interp_type : type of interpolation (default: "linear")
    
    Returns
    -------
    resampled : CTP array resampled to a certain time resolution
    new_times : new time points obtained after scan resampling

    """

    resampled_arrays, resampled_times, time_frames = [], [] , [] 

    for i in range(ctp_array.shape[1]):
        new_times = np.arange(times[i].min(), times[i].max(), delta_t)

        # Apply resampling to the different slices in the Z direction 
        interpolator = interp1d(times[i], ctp_array[:,i], kind=interp_type, axis=0, fill_value="extrapolate")
        resampled = interpolator(new_times)
        resampled_arrays.append(resampled)
        resampled_times.append(new_times)
        time_frames.append(resampled.shape[0])

    # Determine if all slices are resampled to the same number of time frames
    equal_time_frames = len(set(time_frames)) == 1 
    if not(equal_time_frames):
        min_frame = min(time_frames)
        for i in range(ctp_array.shape[1]):
            if time_frames[i] > min_frame:
                # Crop the time frames obtained
                resampled_times[i] = resampled_times[i][:min_frame]
                resampled_arrays[i] = resampled_arrays[i][:min_frame]     


    resampled_arrays = np.stack(resampled_arrays)
    resampled_arrays = np.swapaxes(resampled_arrays, 0, 1)
    resampled_times = np.stack(resampled_times)

    return resampled_arrays, resampled_times


def apply_weighted_moving_average(scan : np.ndarray, time_points : np.ndarray, window_size : int = 3):
    """
    Apply moving average to reduce noise in frames of CTP scan

    Params
    ------
    scan : CTP scan
    time_points : time points for CTP scan
    window_size : window size for moving average

    Returns
    -------
    smoothed_scan : weighted average scan
    
    """
    
    half_window = window_size // 2
    smoothed_scan = scan.copy() # Initialize smoothed scan with a copy of the original
    
    # Pad the time dimension to handle edges
    padded_scan = np.pad(scan, ((half_window, half_window), (0, 0), (0, 0), (0, 0)), mode='edge')
    padded_time_points = np.pad(time_points, (half_window, half_window), mode='edge')
    
    # Apply weighted moving average
    for t in range(scan.shape[0]):
        # Extract the local window of time points
        window = padded_scan[t:(t + window_size)]
        window_times = padded_time_points[t:(t + window_size)]

        # Compute the time differences as weights
        time_diffs = np.diff(window_times)
        if time_diffs.shape[0] > 0:
            time_diffs = np.insert(time_diffs, 0, time_diffs[0])  # Prepend to match window size

            window_sum = time_diffs.sum() # Determine if we are in an edge

            if window_sum > 0:
                ind_non_zero = np.where(time_diffs > 0.)[0]

                if ind_non_zero.shape[0] > 1: # If there is only one non-zero index, do not smooth

                    if ind_non_zero.shape[0] < window.shape[0]:
                        window = window[ind_non_zero]
                        window_times = window_times[ind_non_zero]
                        time_diffs = time_diffs[ind_non_zero]
                
                    
                    #time_diffs = np.insert(time_diffs, 0, time_diffs[0])  # Prepend to match window size
                    

                    #if time_diffs.shape[0] != window_size:
                    #    time_diffs = np.concatenate([time_diffs, np.array([time_diffs[-1]]*abs(window_size-time_diffs.shape[0]))])
                    #else:
                    #    time_diffs = np.append(time_diffs, time_diffs[-1])  # Keep weights consistent
                    
                    # Normalize the weights
                    weights = time_diffs / np.sum(time_diffs)
                    
                    # Apply the weighted average for the current time frame
                    smoothed_scan[t] = np.sum(window * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)

    return smoothed_scan


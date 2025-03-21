import os,sys
import numpy as np
import SimpleITK as sitk
import argparse
from loguru import logger
from typing import Union
import matplotlib.pyplot as plt
import torch

from utils.load_save import load_data, write_data
from utils.segment_mca_ctp import predictionAlgorithm
from utils.curvature import extract_inflection_points, second_cycle
from utils.time_manager import load_time, extract_ids, extract_time_resolution, resample_time, apply_weighted_moving_average


def extract_ctp_array(folder: os.PathLike) -> np.ndarray:
    """
    Extract CTP array from all .nii.gz files in input folder

    Params
    ------
    folder : folder with CTP information

    Returns
    -------
    ctp_array : array with the different CTP frames

    """
    assert os.path.exists(folder), f"Folder '{folder}' does not exist"

    # Load and iterate through each frame file
    ctp_array = []
    # Sort the files by frame index 
    files = sorted(os.listdir(folder), key=lambda x: int(x.split('_t_')[-1].split('.')[0]))
    cont_img = 0
    for file in files:
        if ".nii.gz" in file:
            full_file = os.path.join(folder, file)
            cont_img += 1
            if cont_img == 1:
                image = sitk.ReadImage(full_file)
                img = sitk.GetArrayFromImage(image)
            else:
                img = sitk.GetArrayFromImage(sitk.ReadImage(full_file))
            ctp_array.append(img)
    ctp_array = np.stack(ctp_array, axis=0)
    return ctp_array, image



def extract_avg_frame(ctp : np.ndarray, image, time: np.ndarray) -> np.ndarray:
    """
    Extract average frame from CTP

    Params
    ------
    ctp : perfusion scan array
    image : simple ITK first frame image with spacing information
    time : time array

    Returns
    -------
    avg_frame : average frame array
    avg_frame_image : image with average frame information
    
    """
    # Obtain average frame excluding the beginning and end frames (exclude 10% of longer and shorter times)
    exclude_t = time.max()*0.1
    time_inds_preserve = np.where((time[time.shape[0]//2] > exclude_t) & (time[time.shape[0]//2] < (time[time.shape[0]//2]-exclude_t)))[0]

    avg_frame = np.mean(ctp[time_inds_preserve], axis=0)
    avg_frame_image = sitk.GetImageFromArray(avg_frame)
    avg_frame_image.CopyInformation(image)

    return avg_frame, avg_frame_image


def trim_ctp(ctp_array : np.ndarray, t : np.ndarray, cutoff: float, info: dict, t_secondary: float = None) -> Union[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Trim CTP scan to only include frames during and after bolus
    arrival

    Params
    ------
    ctp_array : CTP frames
    t : time frames
    cutoff : bolus arrival time to MCA-ICA
    info : dictionary with auxiliary temporal information
    t_secondary : time for a secondary peak

    Returns
    -------
    trimmed_ctp : trimmed CTP scan for bolus arrival time
    trimmed_time : time vector trimmed by computed cutoff
    info : updated temporal information dictionary
    bolus_inds : indexes of CTP array kept after trimming

    """
    # Derive bolus indexes
    bolus_inds = np.where(t > cutoff)[0]

    if t_secondary is not None:
        if t_secondary > cutoff:
            # The secondary peak should come after the major, 
            # else we have done something wrong 
            secondary_inds = np.where(t < t_secondary)[0] 
            # Intersect the original bolus indexes with the secondary indexes 
            bolus_inds = np.intersect1d(bolus_inds, secondary_inds)
    else:
        t_secondary = "nan"

    assert (bolus_inds.min() > 0) and (bolus_inds.max() < ctp_array.shape[0]), "Bolus indexes are negative or exceed the number of frames of the CTP scan"

    trimmed_ctp = ctp_array[bolus_inds]
    trimmed_time = t[bolus_inds]

    # Store time frame values considered in the final preprocessing analysis, same file as AIF 
    if info is not None:
        info["trimmed_times"] = trimmed_time.tolist() 

    logger.info(f"Conserved times : {trimmed_time} ")
      

    trimmed_time -= trimmed_time.min()


    return trimmed_ctp, trimmed_time, info, bolus_inds


def extract_cids(ctp_folder : os.PathLike, time_info : dict) -> np.ndarray:
    """
    Extract IDs of cases
    
    Params
    ------
    ctp_folder : folder with CTP data
    time_info : time information with case IDs

    Returns
    -------
    cids : case IDs
     
    """
    # Extract case IDs 
    cids = np.array(sorted(os.listdir(ctp_folder)), dtype=str)

    # Keep only those case IDs with temporal information 
    cids_time = np.array(list(time_info.keys()), dtype=str)

    cids = np.intersect1d(cids, cids_time)

    return cids



def extract_aif(ctp : np.ndarray, mask : np.ndarray) -> Union[np.ndarray, int]:
    """
    Derive Arterial Input Function (AIF) for a certain CTP array,
    given a segmentation of the MCA-ICA

    Params
    ------
    ctp : CTP array
    mask : segmentation for ICA-MCA

    Returns
    -------
    aif : AIF
    bottom_slice : slice of reference where AIF is extracted
    
    """

    # Get the slice with the bottom mask information (earliest flow achievable possible)
    slice_info = np.where(mask > 0)[0]
    bottom_info = np.unique(slice_info)

    # Select as bottom slice the 10th percentile slice
    bottom_slice = int(np.round(np.percentile(bottom_info, 25)))
    logger.info(f"Take segmentation for AIF in slice {bottom_slice}")

    # Consider only the bottom slice
    ctp_bottom_slice = ctp[:, bottom_slice]

    # Create a boolean mask for the bottom slice
    bottom_mask = mask[bottom_slice] > 0

    # Apply the mask to the bottom values
    masked_bottom_values = ctp_bottom_slice[:, bottom_mask]  # Shape: (t, number_of_positive_elements_in_bottom_slice)

    # Compute the mean for each frame
    aif = np.mean(masked_bottom_values, axis=1)

    """
    # Flatten the mask for easier indexing
    masked_array_4d = ctp[:, mask > 0]  # Shape: (t, number_of_positive_mask_elements)

    # Compute the 5th and 95th percentiles for each frame
    lower_bound = np.percentile(masked_array_4d, 5, axis=1)
    upper_bound = np.percentile(masked_array_4d, 95, axis=1)

    # Use broadcasting to filter values within the bounds
    filtered_values = np.where(
        (masked_array_4d >= lower_bound[:, None]) & (masked_array_4d <= upper_bound[:, None]),
        masked_array_4d, 
        np.nan  # We will ignore NaN values when calculating the mean
    )

    # Compute the mean of the filtered values for each frame
    aif = np.nanmean(filtered_values, axis=1)
    """

    return aif, bottom_slice


def case_analysis(cid : str, folder : os.PathLike, output : os.PathLike, cfg : dict, t : np.ndarray, delta_t : float):
    """
    Analyze CTP data of a given case ID

    Params
    ------
    cid : case ID of interest
    folder : folder with CTP information
    output : output folder
    cfg : preprocessing configuration
    t : corresponding time vector for case of interest
    delta_t : time resolution
    
    """
    # Initialize time and AIF information dictionary
    info_file = os.path.join(os.path.dirname(output), "info", f"{cid}.json")
    if os.path.exists(info_file):
        info = load_data(info_file)
    else:
        info = {}
        # info["original_times"] = t.tolist()   

    # CTP loading and derivation of average frame for ICA/MCA derivation
    ctp_array,image = extract_ctp_array(folder = os.path.join(folder,cid))
    avg_frame, avg_frame_image = extract_avg_frame(ctp=ctp_array, image=image, time=t)
    
    # MCA-ICA segmentation with TopCoW24 trained model, for bolus alignment
    # Check first if the AIF has already been computed before
    out_mca = None # Initialize segmentation of the ICA and MCA arteries with TopCoW model 
    aif_slice = ctp_array.shape[1] // 2 # Default slice from where AIF is extracted, if not provided otherwise 
    if not(os.path.exists(info_file)): 
        assert "mca_cpt" in list(cfg.keys()), f"'mca_cpt' key is unavailable in configuration"
        train_dir = cfg["mca_cpt"]
        assert ("tile_step_size" in list(cfg.keys())) and ("use_gaussian" in list(cfg.keys())) and ("use_mirroring" in list(cfg.keys())), f"'tile_step_size', 'use_gaussian', or 'use_mirroring' not in configuration"
        mca,_ = predictionAlgorithm(train_dir=train_dir, device=torch.device("cuda",0), folds=(0,1,2,3,4), tile_step_size=cfg["tile_step_size"], use_gaussian=bool(cfg["use_gaussian"]), use_mirroring=bool(cfg["use_mirroring"])).predict(image_ct=avg_frame_image)

        # MCA-ICA mask is composed by output labels 4, 5, 6, and 7
        ica_segm = (mca == 4).astype(float) + (mca == 6).astype(float)
        out_mca = (ica_segm > 0).astype(float)

        if out_mca.sum() == 0:
            logger.info("No segmentation was found for ICA, trying with MCA...")
            mca_segm = (mca == 5).astype(float) + (mca == 7).astype(float)
            out_mca = (mca_segm > 0).astype(float)

        if out_mca.sum() == 0:
            logger.info("No segmentation was found for ICA nor MCA, trying with the rest of the vessels...")
            out_mca = (mca > 0).astype(float)

        # AIF derivation
        aif, aif_slice = extract_aif(ctp=ctp_array, mask=out_mca)

        # Store AIF
        info["original_aif"] = aif.tolist() 

    else:
        # Load existing AIF
        logger.info(f"Loading existing AIF for case '{cid}'")
        aif = np.array(info["original_aif"])
        

    logger.info(f"AIF extracted, values: {aif}")
    
    if os.path.exists(info_file):
        # Derive cutoff and secondary times from a previous execution, saved as JSON 
        cutoff = info["cutoff"] 
        t_secondary = info["t_secondary"]
        if isinstance(t_secondary, str): # Secondary time is not a number, hence it is not available
            t_secondary = None
    else:
        # Derive cutoff time where the bolus starts coming into 
        # the arteries with inflection points from AIF
        cutoff = extract_inflection_points(x = t[aif_slice], y = aif)
        logger.info(f"Cutoff time: {cutoff} sec")
        # Derive if there exists secondary flow. 
        # If so, restrict the analysis before this secondary flow 
        t_secondary = second_cycle(x = t[aif_slice] , y = aif, 
                                thr_prominence=cfg["thr_prominence"])
        
    if t_secondary is not None:
        logger.info(f"Found secondary flow with peak at time: {t_secondary} sec")


    # Trim CTP array to only include frames after bolus arrival
    # and before any secondary flow 
    ctp_array, t_trim, info, trimming_inds = trim_ctp(ctp_array=ctp_array, t=t[aif_slice], 
                            cutoff=cutoff,
                            info=info,
                            t_secondary=t_secondary)
    logger.info(f"Frames after trimming: {ctp_array.shape[0]}")
    logger.info(f"Time after trimming: {t_trim}")
    
    # Resample to a fixed time resolution
    assert "time_interp" in list(cfg.keys()), "'time_interp' key not in configuration"
    trimmed_time = t[:,trimming_inds] 
    resampled, resampled_times = resample_time(ctp_array=ctp_array, 
                              times=trimmed_time, 
                              delta_t=delta_t,
                              interp_type=cfg["time_interp"])
    logger.info(f"Shape of resampled scan: {resampled.shape}")
    
    # Denoise with moving average techniques
    assert "window" in list(cfg.keys()), "'window' key not in configuration"
    assert ctp_array.shape[0] > cfg["window"], "Trimmed and resampled CTP scans has less frames than the window specified for averaging"
    smoothed = apply_weighted_moving_average(scan=resampled,
                                             time_points=resampled_times[resampled_times.shape[0]//2] ,
                                             window_size=cfg["window"])
    logger.info(f"Shape of smoothed scan: {resampled.shape}")

    # Store AIF and time-related information 
    # if not(os.path.exists(info_file)):
    info["cutoff"] = cutoff
    if t_secondary is None:
        info["t_secondary"] = "nan"
    else:
        info["t_secondary"] = t_secondary 
    
    # info["resampled_times"] = resampled_times.tolist() 
    # Store resampled time information
    resampled_time_folder = os.path.join(os.path.dirname(output), "ctp_time")
    if not(os.path.exists(resampled_time_folder)):
        os.makedirs(resampled_time_folder)
    resampled_time_file = os.path.join(resampled_time_folder, f"{cid}_AcquisitionDateTime.npy")
    np.save(resampled_time_file, resampled_times)

    # Save final AIF curve for resampled and smoothed scan
    if out_mca is not None:
        # AIF derivation
        aif_final, aif_final_slice = extract_aif(ctp=smoothed, mask=out_mca)
        info["final_aif"] = aif_final.tolist()

    # Save final information file
    write_data(data = info, filename=info_file)

    # Set up folder where to store preprocessed CTP data 
    output_cid = os.path.join(output, cid)
    if not(os.path.exists(output_cid)):
        os.makedirs(output_cid)

    for sm in range(smoothed.shape[0]): 
        if sm < 10: 
            outfile = os.path.join(output_cid, f"{cid}_t_0{sm}.nii.gz")
        else:
            outfile = os.path.join(output_cid, f"{cid}_t_{sm}.nii.gz")
        # Avoid reading errors in ITK-SNAP... with np.float32 type
        smoothed_frame = sitk.GetImageFromArray(smoothed[sm].astype(np.float32))   
        smoothed_frame.CopyInformation(image)
        sitk.WriteImage(smoothed_frame, outfile)


def main(args):
    # Preprocess CTP data folder

    # Read arguments
    folder = args.folder # Folder with CTP data subfolders
    time_folder = args.time # Folder with time array data
    out_folder = args.out # Output folder with results

    assert os.path.exists(os.path.dirname(out_folder)), f"Parent folder of output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Set up logfile
    logfile = os.path.join(os.path.dirname(out_folder),"preprocessing.log")
    logger.add(logfile, level="INFO")

    # Prepare folder where to store AIFs, too, to avoid recomputations with GPU
    info_folder = os.path.join(os.path.dirname(out_folder), "info")
    if not(os.path.exists(info_folder)):
        os.makedirs(info_folder)

    # Load config
    assert os.path.exists(os.path.join(os.getcwd(),"config.json")), "Configuration file does not exist"
    cfg = load_data(filename="config.json")

    # Load time resolutions and IDs
    assert "time_delta_default" in list(cfg.keys()), "Key 'time_delta_default' missing in configuration"
    times, delta_t = extract_time_resolution(folder = folder, 
                                             time_folder=time_folder, 
                                             default_delta=cfg["time_delta_default"])
    logger.info(f"Time resolution: {delta_t} sec")
    cids = list(times.keys())

    # Extract case IDs from CTP image folders
    cids = extract_cids(ctp_folder=folder, time_info=times) 

    assert "overwrite" in list(cfg.keys()), f"'overwrite' key not in configuration"

    for cid in cids:
        out_folder_cid = os.path.join(out_folder, cid)
        if not(os.path.exists(out_folder_cid)) or (cfg["overwrite"].lower() != "n"): 
            logger.info(f"Processing case {cid}")
            # If case has already been processed and overwrite is set to "n", skip 
            assert cid in list(times.keys()), f"Case ID '{cid}' not in time information dictionary" 
            case_analysis(cid=cid, folder=folder, cfg = cfg, t=times[cid], output=out_folder, delta_t=delta_t)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder with CTP data", required=True, type=str)
    parser.add_argument("--time", help="Folder with time data", required=True, type=str)
    parser.add_argument("--out", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
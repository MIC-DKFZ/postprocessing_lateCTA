import os,sys
import numpy as np
import SimpleITK as sitk
import argparse
from loguru import logger

from utils.load_save import load_data
from utils.segment_mca_ctp import predictionAlgorithm
from utils.curvature import extract_inflection_points
from utils.time_manager import load_time, extract_ids, extract_time_resolution


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
    files = os.listdir(folder)
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
    # Obtain average frame excluding the begining and end frames (exclude 10% of longer and shorter times)
    exclude_t = time[-1]*0.1
    time_inds_preserve = np.where((time > exclude_t) & (time < (time[-1]-exclude_t)))[0]

    avg_frame = np.mean(ctp[time_inds_preserve], axis=0)
    avg_frame_image = sitk.GetImageFromArray(avg_frame)
    avg_frame_image.CopyInformation(image)

    return avg_frame, avg_frame_image


def trim_ctp(ctp_array : np.ndarray, t : np.ndarray, cutoff: float) -> np.ndarray:
    """
    Trim CTP scan to only include frames during and after bolus
    arrival

    Params
    ------
    ctp_array : CTP frames
    t : time frames
    cutoff : bolus arrival time to MCA-ICA

    Returns
    -------
    trimmed : trimmed CTP scan for bolus arrival time

    """
    # Derive bolus indexes
    bolus_inds = np.where(t > cutoff)[0]
    assert (bolus_inds.min() > 0) and (bolus_inds.max() < ctp_array.shape[0]), "Bolus indexes are negative or exceed the number of frames of the CTP scan"

    return ctp_array[bolus_inds:] 



def extract_aif(ctp : np.ndarray, mask : np.ndarray) -> np.ndarray:
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

    return aif


def case_analysis(cid : str, folder : os.PathLike, cfg : dict, t : np.ndarray):
    """
    Analyze CTP data of a given case ID

    Params
    ------
    cid : case ID of interest
    folder : folder with CTP information
    cfg : preprocessing configuration
    t : corresponding time vector for case of interest
    
    """

    # CTP loading and derivation of average frame
    ctp_array,image = extract_ctp_array(folder = os.path.join(folder,cid))
    avg_frame, avg_frame_image = extract_avg_frame(ctp=ctp_array, image=image, time=t)
    sitk.WriteImage(avg_frame_image, f"{cid}_sum.nii.gz")
    
    # MCA-ICA segmentation with TopCoW24 trained model, for bolus alignment
    assert "mca_cpt" in list(cfg.keys()), f"'mca_cpt' key is unavailable in configuration"
    train_dir = cfg["mca_cpt"]
    mca,_ = predictionAlgorithm(train_dir=train_dir).predict(image_ct=avg_frame_image)

    # MCA-ICA mask is composed by output labels 4, 5, 6, and 7
    mca1 = (mca > 4).astype(float)
    mca2 = (mca < 8).astype(float)
    out_mca = (mca1*mca2).astype(float)
    out_mca_image = sitk.GetImageFromArray(out_mca)
    out_mca_image.CopyInformation(avg_frame_image)
    sitk.WriteImage(out_mca_image, f"{cid}_mca.nii.gz")

    # AIF derivation
    aif = extract_aif(ctp=ctp_array, mask=out_mca)
    logger.info(f"AIF extracted, values: {aif}")

    # Derive cutoff time where the bolus starts coming into 
    # the arteries with inflection points from AIF
    cutoff = extract_inflection_points(x = t, y = aif)
    logger.info(f"Cutoff time: {cutoff} sec")

    np.save("t.npy",t)
    np.save("aif.npy",aif)

    # Trim CTP array to only include frames after bolus arrival
    #ctp_array = trim_ctp(ctp_array=ctp_array, t=t, cutoff=cutoff)



def main(args):
    # Preprocess CTP data folder

    # Read arguments
    folder = args.folder # Folder with CTP data subfolders
    time_folder = args.time # Folder with time array data

    # Load config
    assert os.path.exists(os.path.join(os.getcwd(),"config.json")), "Configuration file does not exist"
    cfg = load_data(filename="config.json")

    # Load time resolutions and IDs
    times, delta_t = extract_time_resolution(folder = folder, time_folder=time_folder)
    cids = list(times.keys())

    for cid in cids:
        logger.info(f"Processing case {cid}")
        case_analysis(cid=cid, folder=folder, cfg = cfg, t=times[cid])



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder with CTP data", required=True, type=str)
    parser.add_argument("--time", help="Folder with time data", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
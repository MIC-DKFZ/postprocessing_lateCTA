import os,sys
import numpy as np
import SimpleITK as sitk
import argparse
from loguru import logger
from utils.load_save import load_data
from utils.segment_mca_ctp import predictionAlgorithm
import matplotlib.pyplot as plt
from typing import Union


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


def extract_avg_frame(ctp : np.ndarray, image) -> np.ndarray:
    """
    Extract average frame from CTP

    Params
    ------
    ctp : perfusion scan array
    image : simple ITK first frame image with spacing information

    Returns
    -------
    avg_frame : average frame array
    avg_frame_image : image with average frame information
    
    """

    avg_frame = np.mean(ctp, axis=0)
    avg_frame_image = sitk.GetImageFromArray(avg_frame)
    avg_frame_image.CopyInformation(image)

    return avg_frame, avg_frame_image



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

    return aif


def case_analysis(cid : str, time_folder : os.PathLike, folder : os.PathLike, cfg : dict):
    """
    Analyze CTP data of a given case ID

    Params
    ------
    cid : case ID of interest
    time_folder : folder with time files
    folder : folder with CTP information
    cfg : preprocessing configuration

    
    """
    # Time loading
    t = load_time(time_folder=time_folder, cid=cid)

    # CTP loading and derivation of average frame
    ctp_array,image = extract_ctp_array(folder = os.path.join(folder,cid))
    avg_frame, avg_frame_image = extract_avg_frame(ctp=ctp_array, image=image)
    
    # MCA-ICA segmentation with TopCoW24 trained model, for bolus alignment
    assert "mca_cpt" in list(cfg.keys()), f"'mca_cpt' key is unavailable in configuration"
    train_dir = cfg["mca_cpt"]
    mca,_ = predictionAlgorithm(train_dir=train_dir).predict(image_ct=avg_frame_image)

    # MCA-ICA mask is composed by output labels 4, 5, 6, and 7
    mca1 = (mca > 4).astype(float)
    mca2 = (mca < 8).astype(float)
    out_mca = (mca1*mca2).astype(float)

    # AIF derivation
    aif = extract_aif(ctp=ctp_array, mask=out_mca)


    plt.figure()
    plt.plot(t, aif)
    plt.title("AIF")
    plt.savefig("aif.png")

    


def main(args):
    # Preprocess CTP data folder

    # Read arguments
    folder = args.folder # Folder with CTP data subfolders
    time_folder = args.time # Folder with time array data

    # Load config
    assert os.path.exists(os.path.join(os.getcwd(),"config.json")), "Configuration file does not exist"
    cfg = load_data(filename="config.json")

    # Load IDs
    cids = extract_ids(folder = folder)

    for cid in cids:
        logger.info(f"Processing case {cid}")
        case_analysis(cid=cid, time_folder=time_folder, folder=folder, cfg = cfg)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder with CTP data", required=True, type=str)
    parser.add_argument("--time", help="Folder with time data", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
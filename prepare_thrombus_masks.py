import os,sys
import SimpleITK as sitk
import numpy as np
import argparse
from typing import Union

from utils.load_save import load_data
from registration.setParameters import set_parameters
from registration.registration_utils import get_transformation_matrix,transform_point


def obtain_cids(folder : os.PathLike) -> list:
    """
    Derive case IDs from input folder, only from 
    NOIV and registry sets

    Params
    ------
    folder : input folder with CTA data

    Returns
    -------
    cids : case IDs
    
    """
    files = sorted(os.listdir(folder))
    cids = []
    for file in files:
        if (".nii.gz" in file) and (("noiv" in file) or (file[0] == "R")):
            cids.append(file.replace(".nii.gz", ""))
    return cids


def derive_thrombus_files(in_folder : os.PathLike, cid : str) -> Union[os.PathLike, os.PathLike]:
    """
    Obtain thrmbus files for a specific case ID

    Params
    ------
    in_folder : input folder
    cid : case ID

    Returns
    -------
    moving_file : CTA information file
    thrombus_file : thrombus file
    
    """

    if cid[0].lower() == "r": # Registry ID
        set_folder = os.path.join(in_folder, "registry_nicolab")
        assert os.path.exists(set_folder), f"Set folder of thrombus data '{set_folder}' does not exist"
        moving_file = os.path.join(set_folder, cid, "registered.nii.gz")
        thrombus_file = os.path.join(set_folder, cid, "thrombus_point.txt")
    elif "mrclean_noiv" in cid: # NOIV ID, prioritze manual labels over Nicolab labels
        set_folder = os.path.join(in_folder, "noiv_manual")
        assert os.path.exists(set_folder), f"Set folder of thrombus data '{set_folder}' does not exist"
        moving_file = os.path.join(set_folder, cid.replace("mr_clean_noiv_", ""), "registered.nii.gz")
        thrombus_file = os.path.join(set_folder, cid.replace("mr_clean_noiv_", ""), "manual_thrombus_point.txt")
        if not(os.path.exists(moving_file)) or not(os.path.exists(thrombus_file)):
            # Try to check if there is a Nicolab label for the same ID
            set_folder = os.path.join(in_folder, "noiv_nicolab") 
            assert os.path.exists(set_folder), f"Set folder of thrombus data '{set_folder}' does not exist"
            moving_file = os.path.join(set_folder, cid.replace("mr_clean_noiv_", ""), "results", "thrombus_result", "registered.nii.gz")
            thrombus_file = os.path.join(set_folder, cid.replace("mr_clean_noiv_", ""), "results", "thrombus_result", "thrombus_point.txt")

    return moving_file, thrombus_file


def load_thrombus_point(file : os.PathLike) -> np.ndarray:
    """
    Load thrombus point from TXT file

    Params
    ------
    file : file with thrombus information

    Returns
    -------
    thrombus : thrombus information
    
    """
    with open(file, "r") as f:
        content = f.read().strip()

    # Remove brackets and convert to numpy array
    thrombus = np.fromstring(content[1:-1], sep=",", dtype=int)

    return thrombus

def obtain_extremes_cta(cta : np.ndarray) -> Union[int,int]:
    """
    Derive trimming extremes from CTA scan

    Params
    ------
    cta : input CTA scan

    Returns
    -------
    low_lim : low extreme
    up_lim : up extreme
    
    """
    # Set up low and up limits as default
    low_lim, up_lim = 0, cta.shape[0] - 1 
    # Obtain mean value of CTA axial slices 
    mean_slice = np.mean(cta, axis = (1,2))
    # Get slices with valuable information 
    information_inds = np.where(mean_slice > -1024)[0]
    low_lim, up_lim = information_inds.min(), information_inds.max()
    
    return low_lim, up_lim


def create_mask_point(cta_img : np.ndarray, low_lim : int, high_lim : int, point : np.ndarray, radius : int) -> np.ndarray:
    """
    Create binary mask around a point of interest, with a certain radius

    Params
    ------
    cta_img : CTA fixed image
    low_lim : low limit axial slice
    up_lim : up limit axial slice
    point : point to create mask around
    radius : radius value

    Returns 
    -------
    mask : output mask

    """

    # Create an empty binary mask
    mask = np.zeros(cta_img.shape, dtype=np.uint8)

    # Set the point in the mask to 1
    mask[tuple(point)] = 1

    # If you want a region around the point (e.g., a small sphere), use:
    zz, yy, xx = np.ogrid[:cta_img.shape[0], :cta_img.shape[1], :cta_img.shape[2]]
    dist = np.sqrt((zz - point[0])**2 + (yy - point[1])**2 + (xx - point[2])**2)
    mask[dist <= radius] = 1

    # Remove parts of the mask outside of CTA limits
    trim_mask = np.zeros(mask.shape)
    trim_mask[low_lim:high_lim+1] = 1
    mask *= trim_mask

    return mask 


def main(args):
    in_folder = args.input
    cta_folder = args.cta
    out_folder = args.out

    assert os.path.exists(in_folder), f"Input folder '{in_folder}' does not exist"
    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent folder of output folder '{os.path.dirname(out_folder)}' does not exist"

    # Load registration configuration
    config_file = "register_config.json"
    assert os.path.exists(config_file), f"Configuration file for registration '{config_file}' does not exist"
    cfg = load_data(filename=config_file)
    cfg_keys = list(cfg.keys())

    # Set up registration parameters
    assert "method" in cfg_keys, "Key 'method' absent in configuration"
    assert "default_val" in cfg_keys, "Key 'default_val' absent in configuration"
    assert "metric" in cfg_keys, "Key 'metric' absent in configuration"
    assert "mask_size" in cfg_keys, "Key 'mask_size' absent in configuration"
    
    p = set_parameters(method=cfg["method"], 
                       DefaultPixelValue=cfg["default_val"], 
                       metric=cfg["metric"])

    # Create output folder if it does not exists
    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Obtain case IDs to be analyzed, only from NOIV and registry sets
    cids = obtain_cids(folder=cta_folder) 

    # Derive moving CTA file and thrombus file
    for cid in cids:
        # Set up moving file and thrombus file
        moving_file, thrombus_file = derive_thrombus_files(in_folder = in_folder, cid = cid)
        # Set up output file
        outfile = os.path.join(out_folder, f"{cid}.nii.gz")  # Skip registration and thrombus mask creation, if it has already been processed
        if os.path.exists(moving_file) and os.path.exists(thrombus_file) and not(os.path.exists(outfile)):
            # Load fixed file
            fixed_file = os.path.join(cta_folder, f"{cid}.nii.gz")
            fixed_image = sitk.ReadImage(fixed_file) 
            fixed_img = sitk.GetArrayFromImage(fixed_image)
            # Derive extremes of CTA images
            low_lim, up_lim = obtain_extremes_cta(cta=fixed_img)
            # Load moving file and thrombus file
            moving_image = sitk.ReadImage(moving_file) 
            thrombus = load_thrombus_point(file = thrombus_file)
            print(thrombus, moving_image.GetSize())
            
            # Register moving to fixed CTA images 
            transform_matrix = get_transformation_matrix(fixed=fixed_image, 
                                                    moving=moving_image, 
                                                    clipvalue=[None, None], 
                                                    parameters=p)
            # Obtain transformed points
            thrombus_transformed = transform_point(transform_parameters=transform_matrix,
                                                   point=thrombus)
            print(thrombus_transformed)

            # Obtain mask around transformed thrombus points
            mask_point = create_mask_point(cta_img = fixed_img, 
                                           low_lim=low_lim, 
                                           high_lim=up_lim, 
                                           point=thrombus_transformed,
                                           radius = cfg["mask_size"])
            print(mask_point.shape, mask_point.sum())
            # Save mask information on transformed thrombus point
            mask_image = sitk.GetImageFromArray(mask_point.astype(np.float32))
            mask_image.CopyInformation(fixed_image)
            sitk.WriteImage(mask_image, outfile) 
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Folder with CTA and original thrombus coordinate data", required=True, type=str)
    parser.add_argument("--cta", help="Folder with preprocessed CTA data", required=True, type=str)
    parser.add_argument("--out", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
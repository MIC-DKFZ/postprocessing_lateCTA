import os,sys
import SimpleITK as sitk
import numpy as np
import argparse
from typing import Union
import time
import matplotlib.pyplot as plt

from utils.load_save import load_data, write_data
from registration.setParameters import set_parameters
from registration.registration_utils import get_transformation_matrix, apply_transformation
from registration.qa_register import compare_files


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


def derive_thrombus_files(in_folder : os.PathLike, cid : str) -> Union[os.PathLike, os.PathLike, os.PathLike, str]:
    """
    Obtain thrmbus files for a specific case ID
    and type of label read

    Params
    ------
    in_folder : input folder
    cid : case ID

    Returns
    -------
    moving_file : CTA information file
    thrombus_file : thrombus point file
    thrombus_mask_file : thrombus mask file
    label_type : either "automated" or "manual"
    
    """
    label_type = "automated"
    if cid[0].lower() == "r": # Registry ID
        set_folder = os.path.join(in_folder, "registry_nicolab")
        mask_folder = os.path.join(in_folder, "registry_nicolab_segms")
        # assert os.path.exists(set_folder), f"Set folder of thrombus data '{set_folder}' does not exist"
        moving_file = os.path.join(set_folder, cid, "registered.nii.gz")
        thrombus_file = os.path.join(set_folder, cid, "thrombus_point.txt")
        thrombus_mask_file = os.path.join(mask_folder, cid, "thrombus_segmentation.nii.gz")
    elif "mrclean_noiv" in cid: # NOIV ID, prioritize manual labels over Nicolab labels
        set_folder = os.path.join(in_folder, "noiv_manual")
        #  assert os.path.exists(set_folder), f"Set folder of thrombus data '{set_folder}' does not exist"
        moving_file = os.path.join(set_folder, cid.replace("mrclean_noiv_", ""), "registered.nii.gz")
        thrombus_file = os.path.join(set_folder, cid.replace("mrclean_noiv_", ""), "manual_thrombus_point.txt")
        thrombus_mask_file = os.path.join(set_folder, cid.replace("mrclean_noiv_", ""), "thrombus_mask.nii.gz")
        label_type = "manual"
        if not(os.path.exists(moving_file)) or not(os.path.exists(thrombus_file)):
            # Try to check if there is a Nicolab label for the same ID
            set_folder = os.path.join(in_folder, "noiv_nicolab") 
            assert os.path.exists(set_folder), f"Set folder of thrombus data '{set_folder}' does not exist"
            moving_file = os.path.join(set_folder, cid.replace("mrclean_noiv_", ""), "results", "thrombus_result", "registered.nii.gz")
            thrombus_file = os.path.join(set_folder, cid.replace("mrclean_noiv_", ""), "results", "thrombus_result", "thrombus_point.txt")
            thrombus_mask_file = os.path.join(set_folder, cid.replace("mrclean_noiv_", ""), "results", "thrombus_mask.nii.gz")
            label_type = "automated"
    return moving_file, thrombus_file, thrombus_mask_file, label_type


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
    mask = mask.astype(np.float32)
    mask *= trim_mask.astype(np.float32)

    return mask 


def create_instance(mask : np.ndarray) -> dict:
    """
    Provide instance information from input thrombus mask

    Params
    ------
    mask : thrombus mask information

    Returns
    -------
    info : instance information
    
    """
    info = {} 
    if mask.sum() > 0:
        info = {"1":0} 
    return {"instances" : info}

def apply_window(image : np.ndarray, window_center : float, window_width : float):
    """
    Apply window to image

    Params
    ------
    image : input image
    window_center : window center
    window_width : window width

    Returns
    -------
    windowed : windowed image
    
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / window_width  # Normalize to [0, 1]
    return windowed

def setup_qa(fixed_img : np.ndarray, reg_img : np.ndarray, moving_img : np.ndarray, mask_img : np.ndarray, thrombus_mask : np.ndarray, point : np.ndarray, point_orig : np.ndarray, low_lim : int, high_lim : int, label : str, circle : bool, outfile : os.PathLike, outfile_qa : os.PathLike):
    """
    Store plots of fixed, moving and registered CTA images and
    compute correlation coefficient between images

    Params
    ------
    fixed_img : fixed image
    reg_img : registered image
    moving_img : moving image
    mask_img : mask image
    thrombus_mask : original mask with thrombus information
    point : registered point data
    point_orig : original point data
    low_lim : low limit in axial direction of fixed image
    high_lim : up limit in axial direction of fixed image
    label : type of label created ("automated" or "manual")
    circle : whether registered mask is circular or not
    outfile : output file
    outfile_qa : output file with correlation results

    Returns
    -------
    Plots comparing images during registration process
    
    """
    # Plot registered image only in the limits of the original image 
    reg_copy = np.zeros(reg_img.shape)-1024
    reg_copy[low_lim:(high_lim + 1)] = reg_img[low_lim:(high_lim + 1)]  

    point = point.flatten()# Just in case, flatten dimensions of point of interest 

    # Apply windowing 
    fixed_img_w = apply_window(image = fixed_img,
                               window_center=55,
                               window_width=110)
    reg_copy_w = apply_window(image = reg_copy,
                               window_center=55,
                               window_width=110)
    moving_w = apply_window(image = moving_img,
                            window_center=55,
                            window_width=110)

    plt.figure()
    plt.subplot(5,3,1)
    plt.imshow(fixed_img_w[point[0]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,2)
    plt.imshow(fixed_img_w[:,point[1]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,3)
    plt.imshow(fixed_img_w[:,:,point[2]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,4)
    plt.imshow(reg_copy_w[point[0]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,5)
    plt.imshow(reg_copy_w[:,point[1]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,6)
    plt.imshow(reg_copy_w[:,:,point[2]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,7)
    plt.imshow(mask_img[point[0]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,8)
    plt.imshow(mask_img[:,point[1]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,9)
    plt.imshow(mask_img[:,:,point[2]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,10)
    plt.imshow(moving_w[point_orig[0]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,11)
    plt.imshow(moving_w[:,point_orig[1]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,12)
    plt.imshow(moving_w[:,:,point_orig[2]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,13)
    plt.imshow(thrombus_mask[point_orig[0]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,14)
    plt.imshow(thrombus_mask[:,point_orig[1]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(5,3,15)
    plt.imshow(thrombus_mask[:,:,point_orig[2]], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outfile)

    # Compute correlation coefficient
    r = compare_files(fixed_img, reg_copy)
    print(r, label)
    cid = os.path.basename(outfile).replace(".png", "")
    circle_info = "thrombus"
    if circle:
        circle_info = "circle"

    with open(outfile_qa, "a") as f:
        f.write(f"{cid},{r},{point[0]},{point[1]},{point[2]},{label},{circle_info}\n")
        f.close()


def derive_centroid(mask : np.ndarray) -> np.ndarray:
    """
    Derive centroid from binary mask

    Params
    ------
    mask : input mask

    Returns
    -------
    centroid : output centroid

    """
    voxel_coords = np.argwhere(mask > 0)

    # Compute the centroid as the mean of these coordinates
    centroid = np.round(voxel_coords.mean(axis=0)).astype(int)
    print(centroid)
    return centroid


def main(args):
    in_folder = args.input
    cta_folder = args.cta
    out_folder = args.out

    assert os.path.exists(in_folder), f"Input folder '{in_folder}' does not exist"
    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent folder of output folder '{os.path.dirname(out_folder)}' does not exist"

    # Load registration configuration
    config_file = "register_thrombus_config.json"
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
    outfile_qa = os.path.join(out_folder, "qa.txt")
    for cid in cids:
        # Set up moving file and thrombus file
        moving_file, thrombus_file, thrombus_mask_file, label_type = derive_thrombus_files(in_folder = in_folder, cid = cid)
        # Set up output file
        outfile = os.path.join(out_folder, f"{cid}.nii.gz")  # Skip registration and thrombus mask creation, if it has already been processed
        if os.path.exists(moving_file) and os.path.exists(thrombus_file) and os.path.exists(thrombus_mask_file) and not(os.path.exists(outfile)):
            print(cid)
            init = time.time()
            # Load fixed file
            fixed_file = os.path.join(cta_folder, f"{cid}.nii.gz")
            fixed_image = sitk.ReadImage(fixed_file) 
            fixed_img = sitk.GetArrayFromImage(fixed_image)
            # Derive extremes of CTA images
            low_lim, up_lim = obtain_extremes_cta(cta=fixed_img)
            print(low_lim, up_lim)
            # Load moving file and thrombus file
            moving_image = sitk.ReadImage(moving_file) 
            moving_img = sitk.GetArrayFromImage(moving_image)
            thrombus = load_thrombus_point(file = thrombus_file)
            print(thrombus, moving_image.GetSize())
            
            # Register moving to fixed CTA images 
            transform_matrix = get_transformation_matrix(fixed=fixed_image, 
                                                    moving=moving_image, 
                                                    clipvalue=[None, None], 
                                                    parameters=p)

            # Load thrombus mask  
            thrombus_mask_image = sitk.ReadImage(thrombus_mask_file)
            thrombus_mask = sitk.GetArrayFromImage(thrombus_mask_image)
            registered_mask = apply_transformation(transform_parameters=transform_matrix,
                                                 moving=thrombus_mask_image,
                                                 segmentation=True,
                                                 default_pixel=0)
            registered_mask.CopyInformation(fixed_image)
            registered_mask_img = sitk.GetArrayFromImage(registered_mask)

            circular_mask = False
            if registered_mask_img.sum() == 0:
                # Thrombus content erased during registration, rescuing it from point
                # Instead create a circular mask based on the thrombus point

                # Apply deformation field to a thrombus mask generated around point of interest 
                mask_point_orig = create_mask_point(cta_img = moving_img, 
                                                low_lim=low_lim, 
                                                high_lim=up_lim, 
                                                point=thrombus,
                                                radius = cfg["mask_size"])
                
                mask_point_orig_image = sitk.GetImageFromArray(mask_point_orig)
                mask_point_orig_image.CopyInformation(moving_image)
                registered_mask = apply_transformation(transform_parameters=transform_matrix,
                                                     moving=mask_point_orig_image,
                                                     segmentation=True,
                                                     default_pixel=0)
                registered_mask.CopyInformation(fixed_image)
                registered_mask_img = sitk.GetArrayFromImage(registered_mask)
                circular_mask = True 
            
            
            # Apply transformation to image also
            new_cta = apply_transformation(transform_matrix, 
                                             moving=moving_image, 
                                             segmentation=False, 
                                             default_pixel=cfg["default_val"]) 
            # Save information on registered image and save matplotlib plot
            new_cta_img = sitk.GetArrayFromImage(new_cta)
            new_cta.CopyInformation(fixed_image)
            outfile_reg = outfile.replace(".nii.gz", "_reg.nii.gz")
            sitk.WriteImage(new_cta, outfile_reg)


            # Get as thrombus transformed point the centroid of the registered mask 
            print(registered_mask_img.sum())
            # mask_point = np.zeros(fixed_img.shape)
            if registered_mask_img.sum() > 0:
                thrombus_transformed = derive_centroid(mask=registered_mask_img)
                
                # Obtain mask around transformed thrombus points
                # mask_point = create_mask_point(cta_img = fixed_img, 
                #                             low_lim=low_lim, 
                #                             high_lim=up_lim, 
                #                             point=thrombus_transformed,
                #                             radius = cfg["mask_size"])
                # print(mask_point.shape, mask_point.sum())

                # QA
                # setup_qa(fixed_img=fixed_img, reg_img = new_cta_img, mask_img=mask_point, 
                #         point=thrombus_transformed, low_lim=low_lim, high_lim = up_lim,
                #         label = label_type, outfile=outfile.replace(".nii.gz", ".png"),
                #         outfile_qa=outfile_qa)
                setup_qa(fixed_img=fixed_img, reg_img = new_cta_img, moving_img=moving_img,
                          mask_img=registered_mask_img, thrombus_mask = thrombus_mask, 
                          point=thrombus_transformed,point_orig=thrombus, low_lim=low_lim, 
                          high_lim = up_lim,label = label_type, 
                          outfile=outfile.replace(".nii.gz", ".png"), circle = circular_mask, 
                          outfile_qa=outfile_qa)

            # Save mask information on transformed thrombus point
            # mask_image = sitk.GetImageFromArray(mask_point.astype(np.float32))
            # mask_image.CopyInformation(fixed_image)
            # sitk.WriteImage(mask_image, outfile) 
            sitk.WriteImage(registered_mask, outfile)

            # Obtain also instance information in json file and save it also
            instance_info = create_instance(mask = registered_mask_img)
            instance_file = outfile.replace(".nii.gz", ".json") 
            write_data(data = instance_info, filename=instance_file)

            print(f"Ellapsed time: {time.time()-init}sec")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Folder with CTA and original thrombus coordinate data", required=True, type=str)
    parser.add_argument("--cta", help="Folder with preprocessed CTA data", required=True, type=str)
    parser.add_argument("--out", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(time.time()-t1)
import os,sys
import SimpleITK as sitk
import numpy as np
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Union
import time
import nibabel as nib
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

from utils.load_save import load_data, write_data
from registration.setParameters import set_parameters
from registration.registration_utils import get_transformation_matrix, apply_transformation
from registration.qa_register import compare_files
from utils.segment_carotid_ctp import segment_brain


def prepare_data_gmm(img : np.ndarray, mask : np.ndarray, brain_mask : np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for Gaussian Mixture model clustering,
    including voxel coordinates and HU values

    Params
    ------
    img : input image
    mask : mask with area of interest
    brain_mask : brain mask

    Returns
    -------
    hus : housfield Units
    coords : coordinates
    std_info : preprocessed data for GMM
    
    """
    # Derive image coordinates 
    coords = list(np.where((mask > 0) & (brain_mask > 0)))
    # Derive HUs  
    hus = [img[(mask > 0) & (brain_mask > 0)].flatten()] 
    # Derive joint information 
    info =  np.vstack(coords + hus).T

    if info.shape[0] > 0: 
        # Standardize information if there is available brain information
        std_info = StandardScaler().fit_transform(info) 
        return np.array(hus).squeeze(), np.vstack(coords).T, std_info
    else:
        return None, None, None


def gmm_model(data : np.ndarray, coords : np.ndarray, hus : np.ndarray) -> np.ndarray:
    """
    Gaussian mixture model of input data
    Derive Hounsfield units of voxels in the highest centroid

    Params
    ------
    data : input data
    coords : coordinates where GMM has been applied
    hus : Hounsfield units where GMM has been applied

    Returns
    -------
    out_hus : output Hounsfield units
    
    """
    k = 2
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
    labels = gmm.fit_predict(data)
    unique_labels = np.unique(labels)

    medians = np.array([np.median(hus[labels == l]) for l in unique_labels])
    high_label = unique_labels[np.argmax(medians)] 

    # Obtain HUs and respective coordinates  
    out_hus = hus[labels == high_label] 
    out_coords = coords[labels == high_label] 

    # Consider values between the 95th and the 99th percentiles of the second cluster found, only
    # Make really sure that we are capturing vessel information
    # Avoid at the same time outliers 
    mask_hus = (out_hus < np.percentile(out_hus.flatten(), 99)) & (out_hus >= np.percentile(out_hus.flatten(), 95)) 
    if mask_hus.sum() > 0:
        # filter to include only high values in the ROI, 
        # making sure that we only take vessel values 
        out_coords = out_coords[mask_hus] 
        out_hus = out_hus[mask_hus]  

    return out_hus, out_coords


def phase_derivation(hu_a : float, hu_v : float) -> float:
    """
    Rule-based derivation of CTA phase, see:

    "Rodriguez-Luna, et al (2014). Venous phase of computed tomography 
    angiography increases spot sign detection, but intracerebral hemorrhage 
    expansion is greater in spot signs detected in arterial phase."

    Params
    ------
    hu_a : HU from arteries around ICA-M1
    hu_v : HU from veins around sinus area

    Returns
    -------
    phase : phase label (0-4 from early arterial to late venous)
    
    """
    phase = -1 # Undetermined phase label
    if (hu_a > hu_v) and (hu_v <= 200):
        phase = 0
    elif (hu_a-hu_v >= 100) and (hu_v > 200) and (hu_a >= hu_v):
        phase = 1
    elif (hu_a-hu_v < 100) and (hu_v > 200) and (hu_a >= hu_v):
        phase = 2
    elif (hu_a < hu_v) and (hu_a > 200):
        phase = 3
    elif (hu_a < hu_v) and (hu_a <= 200):
        phase = 4

    return phase

def sitk2nibabel(sitk_img):
    """
    Convert simple ITK image into nibabel image

    Params
    ------
    sitk_img : simple ITK image
    array : correspinding image array

    Returns
    -------
    nib_img : nibabel image

    """
    # Derive simple ITK attributes 
    spacing = np.array(sitk_img.GetSpacing())  # (x, y, z)
    origin = np.array(sitk_img.GetOrigin())    # (x, y, z)
    direction = np.array(sitk_img.GetDirection())  # flat 3x3

    # Reshape direction to 3x3
    direction_matrix = np.array(direction).reshape(3, 3)

    # Compose affine: column-wise multiplication of direction × spacing
    affine = np.eye(4)
    affine[:3, :3] = direction_matrix * spacing  # scale each column
    affine[:3, 3] = origin

    # Reorder array to match NiBabel's (x, y, z) format
    array = sitk.GetArrayFromImage(sitk_img)
    array = np.transpose(array, (2, 1, 0))  # from (z, y, x) to (x, y, z)

    # Create nibabel image
    nib_img = nib.Nifti1Image(array, affine)

    return nib_img

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

def main(args):
    atlas_folder = args.atlas
    cta_folder = args.cta
    outfile = args.out

    assert os.path.exists(atlas_folder), f"Atlas folder '{atlas_folder}' does not exist"
    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)), f"Parent folder of output phase file '{os.path.dirname(outfile)}' does not exist"

    # Load case IDs
    cta_files = sorted(os.listdir(cta_folder))
    cids = np.array([cta_file.replace(".nii.gz", "") for cta_file in cta_files], dtype=str)

    # Skip case IDs that have already been processed
    phase_results = {} 
    if os.path.exists(outfile):
        phase_results = load_data(filename=outfile)
        cids_computed = np.array(list(phase_results.keys()), dtype=str)
        cids = np.setdiff1d(cids, cids_computed)

    # Load QA file
    qa_file = os.path.join(os.path.dirname(outfile), "qa.json")
    qa = {} 
    if os.path.exists(qa_file):
        qa = load_data(qa_file)

    # Iterate through case IDs
    if cids.shape[0]  > 0:

        # Load atlas information
        atlas_file = os.path.join(atlas_folder, "atlas.nii.gz")
        aif_file = os.path.join(atlas_folder, "aif.nii.gz")
        vof_file = os.path.join(atlas_folder, "vof_new.nii.gz") 

        atlas_image = sitk.ReadImage(atlas_file)
        atlas_img = sitk.GetArrayFromImage(atlas_image)
        aif = sitk.GetArrayFromImage(sitk.ReadImage(aif_file))
        vof = sitk.GetArrayFromImage(sitk.ReadImage(vof_file))

        # Load registration configuration
        config_file = "register_atlas_config.json"
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

        # Iterate through IDs 
        for cid in cids:
            print(f"Processing case ID: {cid}")
            ct_file = os.path.join(cta_folder, f"{cid}.nii.gz")
            ct_image = sitk.ReadImage(ct_file)
        
            # Register CTA to atlas image 
            transform_matrix = get_transformation_matrix(fixed=atlas_image, 
                                                            moving=ct_image, 
                                                            clipvalue=[None, None], 
                                                            parameters=p)
            
            new_cta = apply_transformation(transform_matrix, 
                                                    moving=ct_image, 
                                                    segmentation=False, 
                                                    default_pixel=cfg["default_val"]) 
        
            new_cta.CopyInformation(atlas_image)

            # Compute correlation to measure registration score
            new_cta_img = sitk.GetArrayFromImage(new_cta)
            r = compare_files(cta_scan=new_cta_img, ctp_scan=atlas_img)

            # Convert registered image to nibabel to later process it in totalsegmentator 
            new_cta_nib = sitk2nibabel(sitk_img = new_cta)

            # Segment brain, skull and CCA in registered image to only include brain areas 
            # in phase determination
            brain_segm, skull_segm, cca_segm = segment_brain(img=new_cta_nib)
            brain_mask = ((brain_segm + cca_segm) > 0).astype(float)
            brain_mask[skull_segm > 0] = 0 
            
            # Extract vessels from areas of interest for phase computation
            # with Gaussian Mixture Models 
            artery_hu, artery_coords, artery_data = prepare_data_gmm(img = new_cta_img, 
                                                                     mask = aif, 
                                                                     brain_mask = brain_mask)
            vein_hu, vein_coords, vein_data = prepare_data_gmm(img = new_cta_img, 
                                                               mask = vof, 
                                                               brain_mask = brain_mask)
            
            # Windowed image for QA plotting 
            new_cta_img_w = apply_window(image = new_cta_img, 
                                        window_center = 55, 
                                        window_width = 110) 
            
            if (artery_data is not None) and (vein_data is not None):

                # Obtain HUs and coordinates from selected arterial and venous areas
                final_artery_hus, final_artery_coords = gmm_model(data=artery_data, 
                                                                hus=artery_hu, 
                                                                coords=artery_coords)
                final_vein_hus, final_vein_coords = gmm_model(data=vein_data, 
                                                            hus=vein_hu, 
                                                            coords=vein_coords)

                # Prepare images with segmented areas used for phase computation, for QA 
                final_coords = np.vstack([final_artery_coords, 
                                        final_vein_coords])
                segmented_parts = np.zeros(new_cta_img.shape, dtype=bool)
                segmented_parts[final_coords[:,0], final_coords[:,1], final_coords[:,2]] = True

                # Dilate segmented area for better visualization 
                segmented_parts = binary_dilation(segmented_parts, 
                                                iterations=5).astype(float)
                median_coord = np.median(final_coords, 
                                        axis=0).astype(int).squeeze()
                
                

                median_artery, median_vein = np.median(final_artery_hus), np.median(final_vein_hus)

                # Phase computation
                phase = phase_derivation(hu_a = median_artery,
                                        hu_v=median_vein)
                
                qa[cid] = [float(round(r,3)), float(round(median_artery,2)), float(round(median_vein,2)), float(round(phase))]  

                # Store phase results
                phase_results[cid] = phase

            else:
                # Default values in case the brain segmentation in the registered image fails 
                phase_results[cid] = -1
                qa[cid] = [float(round(r,3)), float(-1.), float(-1.), float(-1.)]  
                median_coord = np.array(new_cta_img.shape)//2 # Just plot middle slices  
                
                
            print(qa[cid])

            plt.figure()
            plt.subplot(231)
            plt.imshow(new_cta_img_w[median_coord[0]], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(232)
            plt.imshow(new_cta_img_w[:,median_coord[1]], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.title(qa[cid])
            plt.subplot(233)
            plt.imshow(new_cta_img_w[:,:,median_coord[2]], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(234)
            plt.imshow(segmented_parts[median_coord[0]], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(235)
            plt.imshow(segmented_parts[:,median_coord[1]], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(236)
            plt.imshow(segmented_parts[:,:,median_coord[2]], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(os.path.dirname(outfile), f"{cid}_qa.png"))

            # Store phase results 
            write_data(data = phase_results, 
                        filename=outfile) 
                
            # Store QA results
            write_data(data = qa,
                    filename=qa_file) 



               

def get_args():
    # Prepare validation set for nnDet experiment as .json and .pkl file
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", help="Folder with atlas information", required=True, type=str)
    parser.add_argument("--cta", help="Folder with CTA information", required=True, type=str)
    parser.add_argument("--out", help="Output .json file with phase information", required=True, type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() -t1)
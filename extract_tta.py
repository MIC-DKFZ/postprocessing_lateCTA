import os,sys
import numpy as np
import SimpleITK as sitk
import argparse
from loguru import logger
from scipy.ndimage import binary_opening, binary_erosion
from sklearn.cluster import KMeans
import nibabel as nib
from typing import Union
from skimage.morphology import remove_small_objects
from scipy.stats import median_abs_deviation
from scipy.ndimage import median_filter

from utils.load_save import load_data
from preprocessing import extract_ctp_array
from utils.segment_carotid_ctp import segment_brain, segment_ica


def get_variation_image(array : np.ndarray, image) -> np.ndarray:
    """
    Study coefficient of variation in array of interest

    Params
    ------
    array : input image to obtain variation information

    Returns
    -------
    cv : image with coefficient of variation information
    
    """
    array -= array.min()
    # maximum = np.max(array, axis=0)
    # minimum = np.min(array, axis=0)
    maximum = np.percentile(array, axis=0, q = 90)
    minimum = np.percentile(array, axis=0, q = 10)
    # std = np.std(array, axis=0)
    mean = np.median(array, axis=0)

    cv = (maximum-minimum) /(mean + np.finfo(float).eps)
    
    # cv = np.std(array, axis=0) / (mean + np.finfo(float).eps)
    # Perform 0-1 clipping 
    cv[cv >= 1] = 0 # Set extreme values to zero 
    cv = np.clip(cv,a_min = 0, a_max=1) # Complete clipping 

    # For each frame in the CTP obtain non-background mask, sum them up, 
    # and those regions with HU below -100 (around 900 in current array) can be discarded, 
    # as they are probably non-vessel
    mask = (array > 900).astype(np.float32)
    mask = np.sum(mask, 0)
    mask = (mask == mask.max()).astype(np.float32)


    # Discard from the variation image those regions where not all the body gets aligned
    cv *= mask 

    return cv

def binarize_variation_image(cv_img : np.ndarray, cfg : dict) -> np.ndarray:
    """
    Binarize variation image into zones with high variability (vessels)
    and low variability (the rest)

    Params
    ------
    cv_img : input variation image
    cfg : configuration parameters


    Returns
    -------
    bin_img : binarized image

    """
    
    assert ("n_clusters" in list(cfg.keys())) and ("state" in list(cfg.keys())) and ("opening_iters" in list(cfg.keys())) and ("remove_thr" in list(cfg.keys())), "Keys 'n_clusters' and/or 'state' and/or 'opening_iters' and/or 'remove_thr' not in configuration"
    
    assert cfg["n_clusters"] > 1, "One or less clusters designated for variation image creation"
    
    # Apply k-means thresholding to isolate high variation information
    model = KMeans(n_clusters=cfg["n_clusters"], 
                   random_state=cfg["state"]) 

    info = cv_img.reshape(-1, 1)
     
    ind_keep = np.where(info > 0)[0]
    info = info[ind_keep]  
    model.fit(info)
    # Derive cluster centers 
    centers = model.cluster_centers_.flatten()

    print(centers)

    # Determine outliers: points beyond the last centroid + FWHM 

    # Obtain the size of the lower centroid, 
    # any voxel with a variation > lower centroid is a vessel voxel
    high_ind = np.argsort(centers)[-1] 
    medium_ind = np.argsort(centers)[1] 

    # Labels derived from k-means model 
    labels = model.labels_.flatten()
    info = info.flatten()

    # Compute intensity distance to lower cluster centroid in k-means 
    # dist = np.abs(info[labels == high_ind] - centers[high_ind])
    # Derive full-width half maximum (FWHM) of distances to lower k-means cluster centroid   
    # fwhm = 2*math.sqrt(math.log(2))*dist.std()

    # Set as threshold the FWHM of the distances 
    thr_cv = centers[medium_ind]
    

    logger.info(f"Binarize variation image with threshold value: {thr_cv}")
    
    bin_img = (cv_img > thr_cv).astype(float)


    # Remove isolated high variation clusters through binary opening
    n_iters = cfg["opening_iters"]
    if n_iters > 0:
        bin_img = binary_opening(bin_img.astype(bool), iterations=n_iters).astype(float)

    # Remove small objects that are probably noise, if not removed by opening
    if cfg["remove_thr"] > 0: 
        bin_img = remove_small_objects(bin_img.astype(bool), 
                                       min_size=cfg["remove_thr"]).astype(float)

    return bin_img




def load_time(time_folder : os.PathLike, cid : str) -> np.ndarray:
    """
    Load time information for a certain case ID of interest

    Params
    ------
    time_folder : folder with time information
    cid : case ID of interest

    Returns
    -------
    time_array : loaded time information
    
    """
    info_file = os.path.join(time_folder, f"{cid}_AcquisitionDateTime.npy")
    assert os.path.exists(info_file), f"Time file '{info_file}' does not exist"

    return np.load(info_file)
    #  info = load_data(info_file)
    # assert "resampled_times" in list(info.keys()), f"Key 'resampled times' not in file '{info_file}'"
    # return np.array(info["resampled_times"])


def detect_outliers_mad(data : np.ndarray, threshold : float =3.5):
    """
    Detect outliers in 1D array

    Params
    ------
    data : input data
    threshold : Z score used as threshold
    
    """
    median = np.median(data)
    mad = median_abs_deviation(data)
    modified_z_scores = 0.6745 * (data - median) / (mad + np.finfo(float).eps)
    return np.where(modified_z_scores > threshold)[0]  # Indices of outliers


def determine_if_consecutive(data : np.ndarray) -> bool:
    """
    Estimate if 1D data is consecutive

    Params
    ------
    data : input 1D array

    Returns
    -------
    consecutive : consecutive data
    
    """
    consecutive = False
    if data.shape[0] > 0:
        diff = np.diff(data)
        # Check if the difference between elements is all just 1 
        consecutive = np.all(diff == 1)
    return consecutive


def extract_cta_array(cta_folder : os.PathLike, cid : str) -> np.ndarray:
    """
    Extract CTA array for a case ID of interest

    Params
    ------
    cta_folder : folder with CTA information
    cid : case ID of interest

    Returns
    -------
    cta_array : array with CTA information
    
    """
    cta_file = os.path.join(cta_folder , f"{cid}.nii.gz")
    assert os.path.exists(cta_file), f"CTA file '{cta_file}' does not exist"
    cta_image = sitk.ReadImage(cta_file)
    cta_array = sitk.GetArrayFromImage(cta_image)
    return cta_image, cta_array


def trim_ctp(ctp_array : np.ndarray, cfg : dict) -> Union[np.ndarray, int, int]:
    """
    If slices of CTP frames slightly vary, find the common slices across 
    frames and trim the rest

    Params
    ------
    ctp_array : input CTP
    cfg : parameter configuration

    Returns
    -------
    ctp_array_out : output trimmed CTP
    low_slice : inferior slice found
    high_slice : superior slice found

    """
    assert "ctp_trimming" in list(cfg.keys()), "'ctp_trimming' not in configuration keys"
    trim = cfg["ctp_trimming"]
    assert trim >= 0, f"Trim parameter is negative ({trim})" 

    # Default extreme values
    low_slice, high_slice = 0, ctp_array.shape[1]-1 
     
    # Obtain slices with empty information 
    ctp_array_out = np.zeros(ctp_array.shape) - 1024 # Scan with -1024 values 
    maxima, minima = [],[]  
    for i in range(ctp_array.shape[0]):
        frame = ctp_array[i]         
        mean_slice = np.mean(frame, axis = (1,2))
        # Get slices with valuable information 
        information_inds = np.where(mean_slice > -1024)[0]
        maxima.append(information_inds.max())
        minima.append(information_inds.min())
    # Obtain low and high slices with information
    low_slice, high_slice = max(minima) + trim, min(maxima) - trim
    
    # Determine if there are any additional acquisition artifacts inducing further CTP frame-wise differences
    mask_ctp_array = (ctp_array > -1024).astype(float)

    variation = []
    for i in range(ctp_array.shape[1]):
        var = 0
        all_equal = np.all(mask_ctp_array[:,i] == 0)
        # Estimate differences for CTP information across frames
        if not(all_equal):
            # Normalize variation of slices in masked CTP 
            # with volume for middle slice with information 
            var = np.abs(mask_ctp_array[:,i] - mask_ctp_array[:,(low_slice + high_slice)//2]).sum() / (mask_ctp_array[:,(low_slice + high_slice)//2].sum() + np.finfo(float).eps)
        variation.append(var)
    variation = np.array(variation) 

    outliers = detect_outliers_mad(variation)

    low_slices = outliers[outliers < (low_slice + high_slice)//2]
    high_slices = outliers[outliers > (low_slice + high_slice)//2]

    # Determine extreme slices with outliers
    print(low_slice, high_slice, outliers) 
    if low_slices.shape[0] > 0: 
        # Recompute inferior extremes if there are any outliers 
        low_slice_outlier = low_slices.max()
        low_slice = max([low_slice, low_slice_outlier])

    if high_slices.shape[0] > 0: 
        # Recompute superior extremes if there are any outliers 
        high_slice_outlier = high_slices.min()
        high_slice = min([high_slice, high_slice_outlier])

    if low_slice >= high_slice:
        # Too much trimming, return original scan, low slice as zero and high slice as last slice 
        return ctp_array, 0, ctp_array.shape[1]-1 

    # Further trim the CTP data if there are more outlier slices 
    if high_slice - low_slice == 2:
        return ctp_array, low_slice, high_slice
     
    ctp_array_out[:,low_slice:high_slice] = ctp_array[:,low_slice:high_slice]

    print(low_slice, high_slice)

    return ctp_array_out, low_slice, high_slice



def segment_brain_components(cta_folder : os.PathLike, segm_folder : os.PathLike, image, cid : str) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    """
    Segment different organ components in the brain, on the CTA image

    Params
    ------
    cta_folder : folder with CTA information
    segm_folder : folder where to store resulting segmentations
    image : reference SimpleITK image used to store information
    cid : case ID of interest

    Returns
    -------
    brain_segm : brain segmentation
    skull_segm : skull segmentation
    cca_segm : common carotid artery segmentation
    ica_segm : internal carotid artery segmentation
    
    """

    # Obtain skull and brain segmentation with TotalSegmentator
    skull_file = os.path.join(segm_folder, f"{cid}_skull.nii.gz")
    brain_file = os.path.join(segm_folder, f"{cid}_brain.nii.gz")
    cca_file = os.path.join(segm_folder, f"{cid}_cca.nii.gz")
    ica_file = os.path.join(segm_folder, f"{cid}_ica.nii.gz")
    logger.info("Deriving segmentation of brain components...")
    if not(os.path.exists(skull_file)) or not(os.path.exists(brain_file)) or not(os.path.exists(cca_file)) or not(os.path.exists(ica_file)):
        # Derive skull segmentation from CTA image
        cta_file = os.path.join(cta_folder, f"{cid}.nii.gz")
        try:
            cta_image = nib.load(cta_file)
        except:
            # Save image with SimpleITK if nibabel does not manage to load it 
            cta_image_sitk = sitk.ReadImage(cta_file)
            cta_image_sitk.CopyInformation(image)
            sitk.WriteImage(cta_image_sitk, cta_file)
            cta_image = nib.load(cta_file)
        brain_segm, skull_segm, cca_segm = segment_brain(img = cta_image) 
        ica_segm = segment_ica(img = cta_image)
        skull_segm_image = sitk.GetImageFromArray(skull_segm)
        skull_segm_image.CopyInformation(image)
        brain_segm_image = sitk.GetImageFromArray(brain_segm)
        brain_segm_image.CopyInformation(image)
        cca_segm_image = sitk.GetImageFromArray(cca_segm)
        cca_segm_image.CopyInformation(image)
        cca_segm_image = sitk.GetImageFromArray(cca_segm)
        cca_segm_image.CopyInformation(image)
        ica_segm_image = sitk.GetImageFromArray(ica_segm)
        ica_segm_image.CopyInformation(image)
        sitk.WriteImage(skull_segm_image, skull_file)
        sitk.WriteImage(brain_segm_image, brain_file)
        sitk.WriteImage(cca_segm_image, cca_file)
        sitk.WriteImage(ica_segm_image, ica_file)
    else:
        # Load segmentations from disk 
        skull_segm = sitk.GetArrayFromImage(sitk.ReadImage(skull_file))
        brain_segm = sitk.GetArrayFromImage(sitk.ReadImage(brain_file))
        ica_segm = sitk.GetArrayFromImage(sitk.ReadImage(ica_file))
        cca_segm = sitk.GetArrayFromImage(sitk.ReadImage(cca_file))

    return brain_segm, skull_segm, cca_segm, ica_segm


def derive_tta(ctp_array : np.ndarray, time_array : np.ndarray) -> np.ndarray:
    """
    Obtain Time To Arrival (TTA) image from CTP information

    Params
    ------
    ctp_array : input 4D CTP image
    time_array : corresponding time information for the CTP array of interest

    Returns
    -------
    tta_img : TTA image
    
    """
    # Derive coordinates of maximum indexes reached 
    time_argmax = np.argmax(ctp_array, axis=0)

    # Derive time resolution in preprocessed space: convert argmax to time coordinates
    # Use middle Z slice resolution as output 
    resolution_array = time_array[time_array.shape[0]//2] - time_array[time_array.shape[0]//2].min()
    tta_img = resolution_array[time_argmax]

    return tta_img 


def cta_masking(cta_array : np.ndarray, low_slice : int, high_slice : int, image, out_cta_folder : os.PathLike, cid : str):
    """
    Remove slices in the CTA scan that are missing in the 
    CTP information

    Params
    ------
    cta_array : input CTA scan
    low_slice : inferior slice to be considered
    high_slice : superior slice to be considered
    image : corresponding CTA image in SimpleITK format
    out_cta_folder : folder where to store resulting masked CTA scan
    cid : case ID of interest

    """
    # Get slices in CTP scan that contain information and restrict the CTA
    # information to those slices
    masked_cta = np.zeros(cta_array.shape) - 1024
    masked_cta[low_slice:high_slice] = cta_array[low_slice:high_slice]  

    # Set up output filename and store resulting image
    out_filename = os.path.join(out_cta_folder, f"{cid}.nii.gz") 
    masked_cta_image = sitk.GetImageFromArray(masked_cta.astype(np.float32))
    masked_cta_image.CopyInformation(image)
    sitk.WriteImage(masked_cta_image, out_filename)


def detect_slices_artifacts(img : np.ndarray, low_slice : int, high_slice : int) -> Union[float, float, np.ndarray]:
    """
    Detect artifacts in top and bottom slices

    Params
    ------
    img : input image
    low_slice : previously computed extreme inferior slice
    high_slice : previously computed extreme superior slice

    Returns
    -------
    low_slice : inferior slice where artifacts start
    high_slice : superior slice where artifacts start
    out_img : corrected output image
    
    """
    # Initialize output image
    out_img = np.zeros(img.shape)

    # Standard deviation value for slice 
    std_slice = np.std(img, axis = (1,2))
    outliers = detect_outliers_mad(std_slice)

    # Obtain inferior and superior slices with artifacts 
    low_slices = outliers[outliers < (low_slice + high_slice)//2]
    high_slices = outliers[outliers > (low_slice + high_slice)//2]

    # Determine if extreme slices are consecutive or not 
    low_consecutive = determine_if_consecutive(data=low_slices)
    high_consecutive = determine_if_consecutive(data=high_slices)

    low_slice_new, high_slice_new = 0, img.shape[0]-1
    if low_consecutive:
        low_slice_new = low_slices.max()
    if high_consecutive:
        high_slice_new = high_slices.min()

    if low_slice >= high_slice:
        low_slice_new, high_slice_new = 0, img.shape[0]-1
        return low_slice_new, high_slice_new, img.copy()
    
    if (high_slice_new - low_slice_new == 2): # Too much trimming, almost consecutive extreme fragments
        return low_slice, high_slice, img.copy()
    
    if (high_slice - low_slice == 2):
        return 0, img.shape[0]-1, img.copy() 
    
    low_slice = max([low_slice, low_slice_new])
    high_slice = min([high_slice, high_slice_new])
    
    # Provide corrected image 
    out_img[low_slice : high_slice] = img[low_slice : high_slice] 
    
    return low_slice, high_slice, out_img



def separate_arterial_venous(tta_img : np.ndarray, cfg : dict, image, out : os.PathLike, cid : str):
    """
    Separate arterial from venous information from TTA image and save
    resulting images

    Params
    ------
    tta_img : TTA image
    cfg : parameter configuration
    image : reference image for SimpleITK saving
    out : output folder
    cid : case ID of interest
    
    """
    assert "separation_method" in list(cfg.keys()), "'separation_method' key not in configuration"
    method = cfg["separation_method"]
    # Derive time value where to apply threshold 
    time_values = tta_img.flatten()
    time_values = time_values[time_values > 0] 
    if method.lower().strip() == "percentile":
        logger.info("Artery-vein separation based on percentiles")
        # Separate veins from arteries assuming a certain percentile 
        # of the cerebral vessel volume is venous and the rest is arterial 
        assert "percentile" in list(cfg.keys()), "'percentile' key not in configuration"
        perc = cfg["percentile"]
        # Consider only TTA values in high-variation regions         
        thr_time = time_values.min() + np.percentile(time_values, 100-perc)
    elif method.lower().strip() == "kmeans":
        # Artery-vein separation with k-means 
        logger.info("Artery-vein separation based on k-means")
        assert "n_clusters_separation" in list(cfg.keys()) and "state_separation" in list(cfg.keys()), "'n_clusters_separation' and/or 'state_separation' keys not in configuration"
        assert cfg["n_clusters_separation"] > 1, "One or less clusters designated for artery-vein separation"
        # K means model 
        model_time = KMeans(n_clusters=cfg["n_clusters_separation"], 
                            random_state=cfg["state_separation"])
        time_values = time_values.reshape(-1, 1)
        model_time.fit(time_values)
        # Set time threshold based on the FWHM of the time distances to the lowest scoring cluster 
        centers_time = model_time.cluster_centers_.flatten()
        middle_ind = np.argsort(centers_time)[1] 
        # low_ind = np.argmin(centers_time)
        # labels_time = model_time.labels_.flatten()
        # dist = np.abs(time_values[labels_time == low_ind] - centers_time[low_ind])
        # fwhm = 2*math.sqrt(math.log(2))*dist.std()

        # thr_time = centers_time[low_ind] + fwhm
        thr_time = centers_time[middle_ind] 

    else:
        raise ValueError("Wrong artery-vein separation method. Choose between the following methods: 'percentile' for percentile-based separation, or 'kmeans' for k-means based separation")

    logger.info(f"Time threshold applied: {thr_time} sec")

    # Derive artery and vein segmentations
    tta_mask = (tta_img > 0).astype(float)
    artery_segm = tta_mask * (tta_img < thr_time).astype(np.float32)
    vein_segm = (tta_img >= thr_time).astype(np.float32)

    # Store images
    artery_segm_image = sitk.GetImageFromArray(artery_segm)
    vein_segm_image = sitk.GetImageFromArray(vein_segm) 
    artery_segm_image.CopyInformation(image)
    vein_segm_image.CopyInformation(image)
    artery_file = os.path.join(out, f"{cid}_early.nii.gz")
    vein_file = os.path.join(out, f"{cid}_late.nii.gz")
    sitk.WriteImage(artery_segm_image, artery_file)
    sitk.WriteImage(vein_segm_image, vein_file)


def obtain_extreme_slices(array : np.ndarray, bgd_val : float = 0.0) -> Union[int,int]:
    """
    Obtain extreme slices for array of interest

    Params
    ------
    array : image array
    bgd_val : background value

    Returns
    -------
    low_slice : inferior slice
    high_slice : superior slice

    """
    assert len(array.shape) == 3, f"Input image array is not 3D ({len(array.shape)})"
    # Get mean value of all axial slices 
    mean_val = np.mean(array, axis=(1,2))
    # Obtain slices with information
    info_slices = np.where(mean_val > bgd_val)[0]
    # Obtain extreme slice coordinates 
    low_slice = info_slices.min()
    high_slice = info_slices.max()
    return low_slice, high_slice 

def case_analysis(folder : os.PathLike, time_folder : os.PathLike, skull_folder : os.PathLike, cta_folder : os.PathLike, out : os.PathLike, out_cta : os.PathLike, cid : str, cfg : dict):
    """
    Analyze preprocessed CTP data for case ID of interest

    Params
    ------
    folder : folder with CTP preprocessed data
    time_folder : folder with corresponding time information of the CTP preprocessed data
    skull_folder : folder with skull segmentation data
    cta_folder : folder with CTA data
    out : output folder where to store TTA information found
    out_cta : output folder where to store masked CTA data found
    cid : case ID of interest
    cfg : configuration parameters
    
    """
    # Iterate through frames to load CTP image
    logger.info("Loading CTP scan...")
    ctp_array, image = extract_ctp_array(folder = folder)

    # Trim CTP and identify low and high slices in the axial direction
    logger.info("Trimming CTP scan...")
    ctp_array, low_slice, high_slice = trim_ctp(ctp_array=ctp_array, cfg=cfg)  
    print(low_slice, high_slice)

    # Extract corresponding time information
    logger.info("Loading corresponding time information...")
    time_array = load_time(time_folder=time_folder, cid=cid) 
    assert time_array.shape[1] == ctp_array.shape[0], f"The time information length ({time_array.shape[1]}) does not coincide with the length of the CTP information ({ctp_array.shape[0]})"  

    # Extract average CTP frame  
    # avg_frame, avg_frame_image = extract_avg_frame(ctp=ctp_array, image=image, time=time_array)

    # Load CTA image
    logger.info("Loading CTA scan...")
    cta_image, cta_array = extract_cta_array(cta_folder=cta_folder, cid=cid)

    # Obtain variation image with approximate vessel information 
    # (zones with high time variability in the CTP scan)
    logger.info("Obtain variation image from preprocessed CTP scan...")
    variation_img = get_variation_image(array = ctp_array, image=cta_image)

    assert "segment_skull_brain" in list(cfg.keys()), "Key 'segment_skull_brain' not in configuration keys"
    do_segm_skull_brain = cfg["segment_skull_brain"]

    if do_segm_skull_brain == 1: 
        # Obtain segmentations of brain components (brain, skull, common carotid artery, internal carotid artery) 
        brain_segm, skull_segm, cca_segm, ica_segm = segment_brain_components(cta_folder=cta_folder,
                                                                              segm_folder = skull_folder,
                                                                              image=image,
                                                                              cid=cid)

        # Discard potential skull information from variation image 
        variation_img *= (1 - skull_segm) 

    # Ignore the first and last slices of the variation image
    assert "trim" in list(cfg.keys()), "Key 'trim' not present in configuration"
    trim = cfg["trim"]
    if trim > 0 and (high_slice - low_slice > 2):
        logger.info("Trimming variation image...")
        variation_img[:trim] = 0 # Top trimming 
        variation_img[(-trim):] = 0 # Bottom trimming  

    if do_segm_skull_brain:
        # Restrict variation image to the brain and the carotid arteries
        # Avoid brain edges (prone to contain registration errors)
        assert "erode_brain_iters" in list(cfg.keys()), "Key 'erode_brain_iters' not in configuration"
        erode_brain_iters = cfg["erode_brain_iters"]
        if erode_brain_iters > 0: 
            brain_segm = binary_erosion(brain_segm, 
                                        iterations=erode_brain_iters).astype(np.int32)  
        variation_mask = ((brain_segm + cca_segm + ica_segm) > 0).astype(np.int32)
        variation_img *= variation_mask

    # Detect slices with registration errors
        
    logger.info("Removing any slices with artifacts in the variation image...")
    low_slice, high_slice, variation_img = detect_slices_artifacts(img = variation_img, 
                                                                    low_slice = low_slice, 
                                                                    high_slice = high_slice)
    
    # Median filter variation image, 
    # to avoid granular noise in the final variation image 
    size = cfg["median_filter_size"]
    assert isinstance(size, int) and size > 0, f"Median filter size '{size}' is not integer or negative" 
    variation_img = median_filter(variation_img, size=size)

    # file = "/scratch/amartinezmora/code/test_cv_image.nii.gz"
    # cv_image = sitk.GetImageFromArray(variation_img)
    # cv_image.CopyInformation(cta_image)
    # sitk.WriteImage(cv_image, file)
    # sys.exit()

    # Isolate zones with high variation in the variation image 
    # (pseudo-vessel segmentation image)
    logger.info("Binarizing variation image...")
    bin_var_img = binarize_variation_image(cv_img = variation_img, cfg=cfg) 
    
    

    # Store binary variation image
    bin_var_image = sitk.GetImageFromArray(bin_var_img.astype(np.float32))
    bin_var_image.CopyInformation(cta_image)
    bin_file = os.path.join(out, f"{cid}_bin.nii.gz")
    sitk.WriteImage(bin_var_image, bin_file)

    # Obtain time-to-arrival image 
    logger.info("Deriving TTA image...")
    tta_img = derive_tta(ctp_array=ctp_array, time_array=time_array) 

    # Restrict TTA image to zones of high variation only
    logger.info("Restricting TTA image to zones of high variation...")
    tta_img *= bin_var_img
    
    # Normalizing image
    tta_img_norm = (tta_img - tta_img.min())/(tta_img.max() - tta_img.min())

    # Store TTA image and normalized version
    logger.info("Storing TTA image and normalized version...")
    tta_image = sitk.GetImageFromArray(tta_img.astype(np.float32))
    tta_image.CopyInformation(cta_image)
    out_tta_file = os.path.join(out, f"{cid}.nii.gz")
    sitk.WriteImage(tta_image, out_tta_file) 

    tta_image_norm = sitk.GetImageFromArray(tta_img_norm.astype(np.float32))
    tta_image_norm.CopyInformation(cta_image)
    out_tta_file_norm = os.path.join(out, f"{cid}_norm.nii.gz")
    sitk.WriteImage(tta_image_norm, out_tta_file_norm) 

    # Derive arterial and venous segmentation information from high-variation image
    assert "separation" in list(cfg.keys()), "'separation' key not in configuration"
    do_separation = cfg["separation"]
    if do_separation.lower().strip() == "y":
        logger.info("Separating arterial from venous information...")
        separate_arterial_venous(tta_img = tta_img, cfg=cfg, image=tta_image, 
                                 out=out, cid = cid) 

    # Corresponding CTA images sometimes comprehend a broader FOV 
    # in the axial direction than TTA images dervied from CTP --> mask them 
    assert "cta_mask" in list(cfg.keys()), f"Key 'cta_mask' not in configuration"
    do_cta_mask = cfg["cta_mask"]
    if do_cta_mask == 1: 
        logger.info("Masking CTA slices that are not present in TTA image...")
        # Determine extreme slices where to trim CTA scan
        low_tta, high_tta = obtain_extreme_slices(array = variation_img, 
                                                  bgd_val=0) 
        print(low_tta, high_tta)
        cta_masking(cta_array=cta_array, low_slice=low_tta, high_slice=high_tta, 
                    image=cta_image, out_cta_folder=out_cta, cid=cid) 


def main(args):
    # Read arguments 
    ctp_folder = args.ctp
    cta_folder = args.cta
    time_folder = args.time
    skull_folder = args.skull
    out_cta_folder = args.out_cta
    out = args.out

    assert os.path.exists(ctp_folder), f"Preprocessed CTP folder '{ctp_folder}' does not exist"
    assert os.path.exists(cta_folder), f"Preprocessed CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(time_folder), f"Preprocessed time folder '{time_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_cta_folder)), f"Parent folder of masked CTA folder '{out_cta_folder}' does not exist"
    assert os.path.exists(os.path.dirname(skull_folder)), f"Parent folder of skull segmentation folder '{skull_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out)), f"Parent output folder '{out}' does not exist"

    # Set up configuration file
    cfg_file = os.path.join(os.path.dirname(__file__), "config_tta.json") 
    assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist"
    cfg = load_data(cfg_file)


    if not(os.path.exists(out)):
        # Create output folder if it does not exist 
        os.makedirs(out)


    if not(os.path.exists(skull_folder)):
        # Create skull segmentation folder if it does not exist 
        os.makedirs(skull_folder)

    if not(os.path.exists(out_cta_folder)):
        # Create output folder for CTA data if it does not exist
        os.makedirs(out_cta_folder) 


    # Set up logfile
    logfile = os.path.join(out, "tta_estimation.log")
    logger.add(logfile, level="INFO")

    # Obtain CIDs: subfolders of preprocessing folder
    cids = sorted(os.listdir(ctp_folder))
    # Ensure case IDs are actually subfolders and not just files  
    cids = [cid for cid in cids if os.path.isdir(os.path.join(ctp_folder, cid))]

    assert "overwrite" in list(cfg.keys()), f"'overwrite' key not in configuration keys"

    # Iterate through case IDs    
    for cid in cids:
        outfile = os.path.join(out, f"{cid}.nii.gz")
        if not(os.path.exists(outfile)) or cfg["overwrite"].lower().strip() == "y": 
            logger.info(f"Processing case ID: {cid}")
            cid_folder = os.path.join(ctp_folder, cid)
            case_analysis(folder = cid_folder, time_folder = time_folder, 
                          skull_folder = skull_folder, cta_folder = cta_folder, 
                          out = out, out_cta=out_cta_folder, cid = cid, cfg = cfg)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctp", help="Folder with preprocessed CTP data", type=str)
    parser.add_argument("--cta", help="Folder with CTA data", type=str)
    parser.add_argument("--time", help="Folder with equivalent time data", type=str)
    parser.add_argument("--skull", help="Folder with skull segmentation", type=str)
    parser.add_argument("--out", help="Output folder with TTA images", type=str)
    parser.add_argument("--out_cta", help="Output folder with masked CTA images", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(get_args())
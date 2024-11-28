import os,sys
import numpy as np
import SimpleITK as sitk
import argparse
from loguru import logger
from scipy.ndimage import binary_erosion, binary_opening
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from preprocessing import extract_ctp_array


from utils.segment_carotid_ctp import segment_brain
# from utils.segment_mca_ctp import predictionAlgorithm


def get_variation_image(array : np.ndarray) -> np.ndarray:
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
    maximum = np.max(array, axis=0)
    minimum = np.min(array, axis=0)
    # std = np.std(array, axis=0)
    mean = np.mean(array, axis=0)

    cv = (maximum-minimum) /(mean + np.finfo(float).eps)
    
    # Perform 0-1 clipping 
    cv = np.clip(cv,a_min = 0, a_max=1)
    cv[cv >= 1] = 0 # Set extreme values to zero after clipping  

    return cv


def case_analysis(folder : os.PathLike, model : os.PathLike, out : os.PathLike, cid : str):
    """
    Analyze preprocessed CTP data for case ID of interest

    Params
    ------
    folder : folder with CTP preprocessed data
    cid : case ID of interest
    out : output folder where to store venous information found
    
    """
    # Iterate through frames 
    # Sort the preprocessed frame files 
    files = os.listdir(folder)
    files = [f for f in files if (".nii.gz" in f) and not("_segm" in f)] 
    files = sorted(files, key=lambda x: int(x.split('_t_')[1].split('.')[0]))
    for file in files:
        if ".nii.gz" in file and (not("segm" in file)):
            logger.info(file)
            full_file = os.path.join(folder, file)  
            frame = sitk.ReadImage(full_file)
            # segm, probs = predictionAlgorithm(train_dir=model, folds=(0,), use_gaussian=True, use_mirroring=False, tile_step_size=0.5).predict(image_ct=frame)
            #  segm_image = sitk.GetImageFromArray(segm)
            # outfile = full_file.replace(".nii.gz", "_segm.nii.gz")
            #sitk.WriteImage(segm_image, outfile)   

    sys.exit()

def main(args):
    # Read arguments 
    folder = args.folder
    model = args.model
    out = args.out

    assert os.path.exists(folder), f"Preprocessed CTP folder '{folder}' does not exist"
    assert os.path.exists(model), f"Segmentation model folder '{model}' does not exist"

    if out is not None:
        if not(os.path.exists(out)):
        # Create output folder if it does not exist 
            os.makedirs(out)

        # Set up logfile
        logfile = os.path.join(out,"venous_estimation.log")
        logger.add(logfile, level="INFO")

    # Obtain CIDs: subfolders of preprocessing folder
    cids = sorted(os.listdir(folder))
    # Ensure case IDs are actually subfolders and not just files  
    cids = [cid for cid in cids if os.path.isdir(os.path.join(folder, cid))]

    # Iterate through case IDs    
    for cid in cids:
        cid_folder = os.path.join(folder, cid)
        case_analysis(folder = cid_folder, model=model, out = out, cid= cid)




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder with preprocessed CTP data", type=str)
    parser.add_argument("--model", help="Folder with model for brain vessel segmentation", type=str)
    parser.add_argument("--out", help="Output folder with venous regions", type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # main(get_args())
    file = "/scratch/amartinezmora/raw_data/mrclean_late_30003"
    file_out = "/scratch/amartinezmora/code/mrclean_late_30003_smoothed.npy"
    image_file = "/scratch/amartinezmora/raw_data/mrclean_late_30003/mrclean_late_30003_t_0.nii.gz"
    outfile = "/scratch/amartinezmora/code/mrclean_late_30003_cv_all.nii.gz"
    brain_file = "/scratch/amartinezmora/code/mrclean_late_30003_brain.nii.gz"
    skull_file = "/scratch/amartinezmora/code/mrclean_late_30003_skull.nii.gz"
    vessel_file = "/scratch/amartinezmora/code/mrclean_late_30003_vessel.nii.gz"
    time_file = "/scratch/amartinezmora/code/mrclean_late_30003_time.nii.gz"
    vein_file = "/scratch/amartinezmora/code/mrclean_late_30003_vein.nii.gz"
    artery_file = "/scratch/amartinezmora/code/mrclean_late_30003_artery.nii.gz" 

    if not(os.path.exists(file_out)):
        ctp_data = extract_ctp_array(folder=file)
        np.save(file_out, ctp_data)
    else:
        ctp_data = np.load(file_out)

    image = sitk.ReadImage(image_file)
    spacing = image.GetSpacing()
    resolution = np.prod(np.array(spacing))

    # Define temporal information  
    time_resolution = 1.522

    # Segment brain from first frame of image
    if not(os.path.exists(brain_file)) or not(os.path.exists(skull_file)):
        brain_segm, skull_segm = segment_brain(img_file = image_file) 
        brain_image = sitk.GetImageFromArray(brain_segm)
        brain_image.CopyInformation(image)
        sitk.WriteImage(brain_image, brain_file)
        skull_image = sitk.GetImageFromArray(skull_segm)
        skull_image.CopyInformation(image)
        sitk.WriteImage(skull_image, skull_file)
    else:
        # If brain segmentation exists, load it  
        brain_segm = sitk.GetArrayFromImage(sitk.ReadImage(brain_file))
        skull_segm = sitk.GetArrayFromImage(sitk.ReadImage(skull_file))
    
    # Derive coefficient of variation image
    cv = get_variation_image(array = ctp_data) 

    # Binarize CV image to erode it and remove edge values
    # with issues caused by alignment artifacts through time
    bin_cv = cv > 0.01
    eroded = binary_erosion(bin_cv, iterations=3).astype(float)
    # Derive brain values, to get the maximum CV value in the brain
    cv_brain = cv[brain_segm > 0]
    cv_brain_max = cv_brain.max()
    cv[cv > cv_brain_max] = 0 
    cv[skull_segm > 0] = 0 # Discard skull also   
    cv *= eroded
    cv *= brain_segm

    
    # Save variation image 
    cv_image = sitk.GetImageFromArray(cv)
    cv_image.CopyInformation(image)
    sitk.WriteImage(cv_image, outfile)

    # Apply k-means thresholding to isolate vessel information
    model = KMeans(n_clusters=2, random_state=42)
    info = cv_brain.reshape(-1, 1)
    model.fit(info)
    centers = model.cluster_centers_
    # Consider the voxel values between the centroids and take the median 
    thr = centers.mean()
    vessel_segm = (cv >= thr).astype(float)

    print(vessel_segm.sum(), thr, centers, cv.min(), cv.max())
    sys.exit()

    vessel_image = sitk.GetImageFromArray(vessel_segm)
    vessel_image.CopyInformation(image)
    sitk.WriteImage(vessel_image, vessel_file)

    # Extract peak time information for the different segmented vessels
    time_argmax = np.argmax(ctp_data, axis=0)*time_resolution
    time_segm = vessel_segm*time_argmax

    time_values = time_segm[vessel_segm > 0]
    print(time_values.mean())

    # Estimate approximate number of venous voxels 
    #  (around 70% of the blood in the brain is in veins)   
    # This is equivalent to taking the 30th percentile of the times  
    thr_time = np.percentile(time_values.flatten(), 30)
    

    # model_time = KMeans(n_clusters=2, random_state=41)
    # model_time.fit(time_values)
    # centers_time = model_time.cluster_centers_
    # thr_time = centers_time.mean()

    # Define bin edges by resolution  # Width of each bin
    # bins = np.arange(time_values.min(), time_values.max() + time_resolution, time_resolution)
    # Plot the histogram
    # plt.figure()
    # plt.hist(time_values, bins=bins, color='blue', edgecolor='black', alpha=0.7) 
    # plt.savefig("hist.png") 


    time_image = sitk.GetImageFromArray(time_segm)
    time_image.CopyInformation(image)
    sitk.WriteImage(time_image, time_file)


    vein_segm = time_segm >= thr_time
    artery_segm = (time_segm < thr_time).astype(float)
    vein_segm = binary_opening(vein_segm).astype(float)
    artery_segm *= vessel_segm
    print(artery_segm.sum(), vein_segm.sum())
    vein_image = sitk.GetImageFromArray(vein_segm)
    sitk.WriteImage(vein_image, vein_file)
    artery_image = sitk.GetImageFromArray(artery_segm)
    sitk.WriteImage(artery_image, artery_file)
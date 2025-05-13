import os,sys
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_erosion
import time
import skfmm
import argparse
import nibabel as nib
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from compute_phase import apply_window
from utils.segment_carotid_ctp import segment_brain

def distance_map(img : np.ndarray, val_min : float, val_max : float, spacing : np.ndarray):
    """
    Obtain map of distances for a given mask and its 
    respective spacing

    Params
    ------
    img : input image with time information
    val_min : minimum time value of interest
    val_max : maximum time value of interest
    mask : input mask
    spacing : respective spacing


    Returns
    -------
    inv_dist : map with inverse distances
    
    """
    # Obtain mask from time image, only for value of interest 
    mask = ((img >= val_min) & (img < val_max)).astype(int)

    # Derive minimum spatial distance in map (voxel diagonal) 
    voxel_diagonal = np.linalg.norm(spacing)

    # Convert mask to -1 in foreground, and +1 in background 
    phi = -2*mask + 1
    dist = skfmm.distance(phi, dx=spacing) + voxel_diagonal
    inv_dist = 1/dist

    return inv_dist


def bin_times(img : np.ndarray, bins : int = 4) -> np.ndarray:
    """
    Bin values from time map

    Params
    ------
    img : input time map
    bins : bins for time data


    Returns
    -------
    vals : resulting time values to analyze
    
    """
    bin_mask = img != 0.0 # Obtain binary vasculature image 
    # Erode image with two iterations, to remove time values with little influence and save computation time
    eroded_mask = binary_erosion(bin_mask, iterations=2)  

    # Resulting time values 
    time_values = img[eroded_mask].flatten()
    vals = np.unique(time_values)

    # Determine percentiles where to apply binning
    if vals.shape[0] > bins:  
        ps = np.linspace(0, 100, num=bins+1)
        vals = np.array([np.percentile(vals, p) for p in ps])

    return vals


def weigh_distance_maps(dist_maps : list, vals : np.ndarray, out_file : os.PathLike, image, brain : np.ndarray = None):
    """
    Weigh and combine distance maps for different times

    Params
    ------
    dist_maps : set of distance maps computed for different times
    vals : different time values
    out_file : output file
    image : original time map image
    brain : brain segmentation (default: None)

    Returns
    -------
    final : final combination of time and distance
    
    """
    # convert maps to stack 
    dist_maps = np.stack(dist_maps)

    # Normalize weight maps
    weight_maps = dist_maps/(np.sum(dist_maps,0) + np.finfo(float).eps)

    if vals.shape[0] == (weight_maps.shape[0] + 1):
        # Derive respective time values from bin information
        # Take average values between consecutive time points    
        vals = 0.5*(vals[1:] + vals[:(-1)])  

    # Obtain final map by weighting with time values 
    final = np.sum(vals[:, None, None, None] * weight_maps, axis=0)

    # Remove values outside the brain    
    if brain is not None:
        final[brain == 0] = 0

    # Save final result 
    final_map_image = sitk.GetImageFromArray(final.astype(np.float32))
    final_map_image.CopyInformation(image)
    sitk.WriteImage(final_map_image, out_file)

    return final


def signed_time_image(img : np.ndarray):
    """
    Provide a sign to the time image. Giving negative values to venous areas, 
    and positive values to arterial areas

    Use a percentile of 30 as separation between arteries and veins,
    according to medical literature

    Params
    ------
    img : input time image
    

    Returns
    -------
    signed_img : signed time image

    """
    time_vals = img[img > 0].flatten()
    p30 = np.percentile(time_vals, 30)
    signed_img = np.zeros(img.shape)
    signed_img[img > 0] = p30 + 0.01 - img[img > 0]

    return signed_img    


def main(args):
    time_folder = args.t
    brain_folder = args.b
    workers = args.w
    out_folder = args.o
    bins = args.bin

    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    assert os.path.exists(brain_folder), f"Brain folder '{brain_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        # Create output folder if it does not exist 
        os.makedirs(out_folder)

    # Derive case IDs to compute
    files = sorted(os.listdir(time_folder))
    tag = "_norm.nii.gz" # File tag to look for 

    for file in files:
        if tag in file:
            cid = file.replace(tag, "")
            outfile = os.path.join(out_folder, f"{cid}.nii.gz")

            if not(os.path.exists(outfile)):
                # Skip already processed files
                print(f"Process case ID: {cid}") 

                # Check if brain file exists, otherwise segment the brain
                brain_file = os.path.join(brain_folder, f"{cid}_brain.nii.gz")
                if os.path.exists(brain_file):
                    brain = sitk.GetArrayFromImage(sitk.ReadImage(brain_file))
                else:
                    cta_nib = nib.load(brain_file)
                    brain, _ ,_ = segment_brain(img = cta_nib)

                # Load time image and respective spacing
                time_file = os.path.join(time_folder, file)

                image = sitk.ReadImage(time_file)
                img = sitk.GetArrayFromImage(image)
                spacing = np.array(image.GetSpacing())

                # Obtain signed image providing different signs to arteries and veins
                # signed_img = signed_time_image(img = img) 

                # Bin time image and get time values of interest
                # time_vals = bin_times(img = signed_img, bins = bins)
                time_vals = bin_times(img = img, bins = bins)

                print(time_vals)

                # Derive maps for different time steps
                # distance_maps = [distance_map(img = signed_img, val = t, spacing = spacing) for t in time_vals]
                # distance_maps = Parallel(n_jobs=workers)(delayed(distance_map)(signed_img, time_vals[i], time_vals[i+1], spacing) for i in range(time_vals.shape[0]-1))
                distance_maps = Parallel(n_jobs=workers)(delayed(distance_map)(img, time_vals[i], time_vals[i+1], spacing) for i in range(time_vals.shape[0]-1))

                # Obtain final image
                final = weigh_distance_maps(dist_maps=distance_maps, 
                                             vals=time_vals, 
                                             out_file=outfile, 
                                             image=image,
                                             brain=brain)
                # final[brain == 0] = 0

                # Save final result 
                # final_map_image = sitk.GetImageFromArray(final.astype(np.float32))
                # final_map_image.CopyInformation(image)
                # sitk.WriteImage(final_map_image, outfile)
                
                # Save plots for QA 
                ww_img = np.percentile(img[img > 0], 99)
                ww_final = np.percentile(final[brain > 0], 99) - np.percentile(final[brain > 0], 1)
                min_final = np.percentile(final[brain > 0], 1) + ww_final/2
                final_w = apply_window(image=final, window_center=min_final, window_width=ww_final)
                img_w = apply_window(image=img, window_center=ww_img/2, window_width=ww_img)
                images = [final_w, img_w] 

                plt.figure()
                for enum_i, i in enumerate(images):
                    plt.subplot(2,3,enum_i*3 + 1)
                    plt.imshow(i[i.shape[0]//2])
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar()
                    plt.subplot(2,3,enum_i*3 + 2)
                    plt.imshow(i[:,i.shape[1]//2])
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar()
                    plt.subplot(2,3,enum_i*3 + 3)
                    plt.imshow(i[:,:,i.shape[2]//2])
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar()

                plt.savefig(outfile.replace(".nii.gz", ".png"))
                  
     


def get_args():
    # Derive time-distance maps for translation model development
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", help="Folder with time information", required=True, type=str)
    parser.add_argument("--b", help="Folder with brain information", required=True, type=str)
    parser.add_argument("--w", help="Number of parallel workers", default=6, type=int)
    parser.add_argument("--bin", help="Number of bins to structure time information", default=4, type=int)
    parser.add_argument("--o", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() -t1)
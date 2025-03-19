import os,sys
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import label
import numpy as np
from typing import Union
import pandas as pd

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


def main(args):
    cta_folder = args.cta
    tta_folder = args.tta
    out_folder = args.out

    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(tta_folder), f"TTA folder '{tta_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent directory of output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Iterate through CTA scans
    cta_files = sorted(os.listdir(cta_folder))
    out_dict = {"cid" : [], "conn_comp" : [], "mean" : [], "std" : [], "median" : [],
                "min" : [], "max" : [], "low_slice_cta" : [], "high_slice_cta" : [],
                "low_slice_tta" : [], "high_slice_tta" : []} 
    
    stats_file = os.path.join(out_folder, "stats_cta_tta.csv")

    for cta_file in cta_files:
        if ".nii.gz" in cta_file:
            # Load CTA and TTA images
            cid = cta_file.replace(".nii.gz", "")
            print(cid)
            out_dict["cid"].append(cid) 
            full_file = os.path.join(cta_folder, cta_file) 
            cta_image = sitk.ReadImage(full_file)
            cta = sitk.GetArrayFromImage(cta_image)
            tta_file = os.path.join(tta_folder, f"{cid}.nii.gz")
            tta_image = sitk.ReadImage(tta_file)
            tta = sitk.GetArrayFromImage(tta_image)

            outfile = os.path.join(out_folder, f"{cid}.png")
            
            plt.figure()
            plt.subplot(231)
            plt.imshow(cta[cta.shape[0]//2], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(232)
            plt.imshow(cta[:,cta.shape[1]//2], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(233)
            plt.imshow(cta[:,:,cta.shape[2]//2], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(234)
            plt.imshow(tta[tta.shape[0]//2], cmap="cool")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(235)
            plt.imshow(tta[:,tta.shape[1]//2], cmap="cool")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(236)
            plt.imshow(tta[:,:,tta.shape[2]//2], cmap="cool")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(outfile)

            # Connected components
            bin_tta = (tta > 0).astype(float) 
            _, num_features = label(bin_tta)
            out_dict["conn_comp"].append(num_features)

            # TTA stats
            tta_vals = tta[tta > 0].flatten()
            out_dict["max"].append(np.percentile(tta_vals, 95))   
            out_dict["min"].append(np.percentile(tta_vals, 5)) 
            out_dict["median"].append(np.median(tta_vals))
            out_dict["mean"].append(tta_vals.mean())
            out_dict["std"].append(tta_vals.std())  

            # Extreme slices   
            low_cta, high_cta = obtain_extreme_slices(array = cta, bgd_val=-1024)
            low_tta, high_tta = obtain_extreme_slices(array = tta, bgd_val=0)

            out_dict["low_slice_cta"].append(low_cta) 
            out_dict["low_slice_tta"].append(low_tta) 
            out_dict["high_slice_cta"].append(high_cta) 
            out_dict["high_slice_tta"].append(high_tta) 


    df = pd.DataFrame(out_dict)
    df.to_csv(stats_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", help="Folder with TTA data", type=str)
    parser.add_argument("--cta", help="Folder with CTA data", type=str)
    parser.add_argument("--out", help="Output folder for QA plots", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(get_args())
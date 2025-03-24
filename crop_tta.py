import os,sys
import SimpleITK as sitk
import argparse
import numpy as np
import pandas as pd
import time

def crop_case(cid : str, cta_folder : os.PathLike, tta_folder : os.PathLike, out_folder : os.PathLike, low_lim : int = -1, high_lim : int = -1):
    """
    Crop case of interest to instructions given and save files

    Params
    ------
    cid : case ID of interest
    cta_folder : CTA folder
    tta_folder : TTA map folder
    out_folder : output folder
    low_lim : lower limit for crop (if no low cropping, -1)
    high_lim : higher limit for crop (if no high cropping, -1)
    
    """
    if low_lim != -1 and high_lim != -1:
        assert low_lim <= high_lim, f"Lower limit ({low_lim}) is not inferior to superior limit ({high_lim})"

    # Access CTA and TTA files 
    cta_file = os.path.join(cta_folder, f"{cid}.nii.gz")
    tta_file = os.path.join(tta_folder, f"{cid}.nii.gz")
    tta_norm_file = os.path.join(tta_folder, f"{cid}_norm.nii.gz")
    tta_bin_file = os.path.join(tta_folder, f"{cid}_bin.nii.gz")
    tta_early_file = os.path.join(tta_folder, f"{cid}_early.nii.gz")
    tta_late_file = os.path.join(tta_folder, f"{cid}_late.nii.gz")

    files = [cta_file, tta_file, tta_norm_file, 
             tta_bin_file, tta_early_file, tta_late_file] 
    
    for enum_file, file in enumerate(files):
        out = os.path.join(out_folder, "tta_crop")
        bgd = 0
        if enum_file == 0:
            # Work with CTA images 
            out = os.path.join(out_folder, "cta_crop")
            bgd = -1024

        if not(os.path.exists(out)):
            os.makedirs(out)

        out_file = os.path.join(out, os.path.basename(file))

        # Read image 
        image = sitk.ReadImage(file)
        img = sitk.GetArrayFromImage(image)
        
        # Assertions
        if low_lim != -1:
            assert low_lim < img.shape[0], "Low cropping limit exceeding image dimensions" 

        if high_lim != -1:
            assert high_lim < img.shape[0], "High cropping limit exceeding image dimensions" 

        # Set up output image 
        out_img = np.zeros(img.shape) + bgd

        # Complete cropping 
        
        if low_lim == -1:
            # Only superior cropping 
            out_img[:high_lim] = img[:high_lim] 

        if high_lim == -1:
            # Only inferior cropping 
            out_img[low_lim:] = img[low_lim:]

        if low_lim != -1 and high_lim != -1:
            # Both inferior and superior cropping
            out_img[low_lim:high_lim] = img[low_lim:high_lim]

        out_image = sitk.GetImageFromArray(out_img)
        out_image.CopyInformation(image)
        sitk.WriteImage(out_image, out_file)



def main(args):
    cta_folder = args.cta
    tta_folder = args.tta
    crop_file = args.crop_info
    out_folder = args.out

    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(tta_folder), f"TTA folder '{tta_folder}' does not exist"
    assert os.path.exists(crop_file), f"Crop file '{crop_file}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent of output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        # Create output folder if it does not exist
        os.makedirs(out_folder)

    # Load file with cropping information
    df = pd.read_csv(crop_file)
    cids = df["cid"].values.astype(str)
    low_lims = df["crop_slice_inferior"].values
    high_lims = df["crop_slice_superior"].values  

    # Iterate through IDs 
    for cid, low_lim, high_lim in zip(cids, low_lims, high_lims):
        crop_case(cid = cid, cta_folder=cta_folder, tta_folder=tta_folder,
                  out_folder=out_folder, low_lim=low_lim, high_lim= high_lim)
     

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cta", help="Folder with preprocessed CTA data", type=str)
    parser.add_argument("--tta", help="Folder with preprocessed TTA maps", type=str)
    parser.add_argument("--crop_info", help="File with cropping information", type=str)
    parser.add_argument("--out", help="Output folder with CTA and TTA data", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time() - t1} sec")
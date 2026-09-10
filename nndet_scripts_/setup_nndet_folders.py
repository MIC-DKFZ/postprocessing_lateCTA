# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os,sys
import shutil
import argparse
import time
import SimpleITK as sitk
import numpy as np

def derive_cids(cta_folder : os.PathLike, time_folder : os.PathLike, thrombus_folder : os.PathLike, process : str = "tta") -> list:
    """
    Derive case IDs of interest: they should be all in the CTA, in the time,
    and in the thrombus folder

    Params
    ------
    cta_folder : folder with CTA data
    time_folder : folder with time data
    thrombus_folder : folder with thrombus data
    process : type of data to be processed as additional channel

    Returns
    -------
    cids : case IDs to be utilized
    
    """
    # Derive IDs from CTA folder
    cta_files = sorted(os.listdir(cta_folder))
    cta_cids = [file.replace(".nii.gz", "") for file in cta_files if ".nii.gz" in file]  

    # Derive IDs from time folder
    time_files = sorted(os.listdir(time_folder))
    bin_ids, norm_ids = [], []

    if (process != "late") and (process != "dist"):
        for time_file in time_files:
            if "_bin.nii.gz" in time_file:
                # ID for time binary file 
                bin_ids.append(time_file.replace("_bin.nii.gz", ""))
            elif "_norm.nii.gz" in time_file:
                # ID for time normalized file 
                norm_ids.append(time_file.replace("_norm.nii.gz", ""))
        time_cids = list(set(bin_ids) & set(norm_ids))
    else:
        time_cids = [f.replace(".nii.gz", "") for f in time_files if ".nii.gz" in f] 


    


    # Derive IDs from thrombus folder
    thrombus_files = sorted(os.listdir(thrombus_folder))
    thrombus_cids = [file.replace(".nii.gz", "") for file in thrombus_files if ".nii.gz" in file] 

    # Derive intersection of the three sets of IDs
    cids = list(set(cta_cids) & set(time_cids) & set(thrombus_cids)) 

    return cids 

    

def main(args):
    time_folder = args.t
    cta_folder = args.cta
    thrombus_folder = args.thrombus
    process = args.process
    out_folder = args.out

    assert os.path.exists(time_folder), f"Time data folder '{time_folder}' does not exist"
    assert os.path.exists(cta_folder), f"CTA data folder '{cta_folder}' does not exist"
    assert os.path.exists(thrombus_folder), f"Thrombus data folder '{thrombus_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent of output data folder '{os.path.dirname(out_folder)}' does not exist"
    assert process.lower().strip() in ["none", "tta", "bin", "late", "dist"], f"Process '{process}' not 'none', nor 'tta', nor 'bin', nor 'late'"

    # Create output folder if it does not exist
    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder) 

    # Determine IDs to be processed: they need to have CTA, time, and thrombus information
    cids = derive_cids(cta_folder=cta_folder, time_folder=time_folder,
                       thrombus_folder=thrombus_folder, 
                       process = process.lower().strip())

    # Set up the tree of output directories
    raw_folder = os.path.join(out_folder, "raw_splitted") 
    image_folder = os.path.join(raw_folder, "imagesTr")
    label_folder = os.path.join(raw_folder, "labelsTr")

    image_folder_test = os.path.join(raw_folder, "imagesTs")
    label_folder_test = os.path.join(raw_folder, "labelsTs")

    if not(os.path.exists(image_folder)):
        # Create image folder if it does not exist 
        os.makedirs(image_folder)

    if not(os.path.exists(label_folder)):
        # Create label folder if it does not exist 
        os.makedirs(label_folder)

    if not(os.path.exists(image_folder_test)):
        # Create image test folder if it does not exist 
        os.makedirs(image_folder_test)

    if not(os.path.exists(label_folder_test)):
        # Create label test folder if it does not exist 
        os.makedirs(label_folder_test)

    for cid in cids: # Iterate through IDs  
        # Transfer CTA data to nnDet det_data structure        
        cta_file = os.path.join(cta_folder, f"{cid}.nii.gz")
        cta_outfile = os.path.join(image_folder, f"{cid}_0000.nii.gz")
        shutil.copyfile(cta_file, cta_outfile)

        # Transfer label data to nnDet det_data structure
        instance_file = os.path.join(thrombus_folder, f"{cid}.json")
        label_file = os.path.join(thrombus_folder, f"{cid}.nii.gz")
        instance_outfile = os.path.join(label_folder, f"{cid}.json")
        label_outfile = os.path.join(label_folder, f"{cid}.nii.gz")
        shutil.copyfile(instance_file, instance_outfile)
        shutil.copyfile(label_file, label_outfile)

        # Transfer time data to nnDet det_data structure
        time_file = None
        if process.lower().strip() == "bin":
            time_file = os.path.join(time_folder, f"{cid}_bin.nii.gz") 
        elif process.lower().strip() == "tta":
            time_file = os.path.join(time_folder, f"{cid}_norm.nii.gz") 
        elif (process.lower().strip() == "late") or (process.lower().strip() == "dist"):
            time_file = os.path.join(time_folder, f"{cid}.nii.gz") 
        time_outfile = os.path.join(image_folder, f"{cid}_0001.nii.gz")

        if time_file is not None:
            # time_image = sitk.ReadImage(time_file)
            # time_img = sitk.GetArrayFromImage(time_image)
            # time_img = np.clip(time_img, 0, 1)
            # time_image_out = sitk.GetImageFromArray(time_img)
            # time_image_out.CopyInformation(time_image)
            # sitk.WriteImage(time_image_out, time_outfile)
            shutil.copyfile(time_file, time_outfile)
        
      

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", help="Folder with preprocessed TTA data", required=True, type=str)
    parser.add_argument("--cta", help="Folder with preprocessed CTA data", required=True, type=str)
    parser.add_argument("--thrombus", help="Folder with preprocessed thrombus data", required=True, type=str)
    parser.add_argument("--process", help="Processing required ('none', 'tta', 'bin', 'late', 'dist')", default="none", type=str)
    parser.add_argument("--out", help="Output folder with data to be processed by nnDetection", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(time.time()-t1)
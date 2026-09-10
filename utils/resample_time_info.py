# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os,sys
import SimpleITK as sitk
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def resample_rows_interp(arr, new_rows, kind='linear'):
    old_x = np.linspace(0, 1, arr.shape[0])
    new_x = np.linspace(0, 1, new_rows)
    interpolator = interp1d(old_x, arr, axis=0, kind=kind, fill_value="extrapolate")
    return interpolator(new_x)

def main(args):
    time_folder = args.time
    cta_folder = args.cta
    out_folder = args.out

    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent of output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Iterate through CTA files
    cta_files = sorted(os.listdir(cta_folder))
    for cta_file in cta_files:
        if ".nii.gz" in cta_file:
            cid = cta_file.replace(".nii.gz","")
            outfile = os.path.join(out_folder, f"{cid}_AcquisitionDateTime.npy")
            if not(os.path.exists(outfile)):
                full_file = os.path.join(cta_folder, cta_file)
                cta = sitk.GetArrayFromImage(sitk.ReadImage(full_file))
                time_file = os.path.join(time_folder, f"{cid}_AcquisitionDateTime.npy")
                if not(os.path.exists(time_file)):
                    time_file = os.path.join(time_folder, f"{cid}_AcquisitionTime.npy")
                
                time_info = np.load(time_file)
            
                # Target number of rows to resample 
                n_rows = cta.shape[0]
                new_time = resample_rows_interp(arr=time_info, new_rows=n_rows)

                np.save(outfile, new_time)


def get_args():
    # Identify those case IDs from the registration process to be removed for further preprocessing 
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", help="Folder with time folder information", required=True, type=str)
    parser.add_argument("--cta", help="Folder with CTA information", required=True, type=str)
    parser.add_argument("--out", help="Output folder", required=True, type=str)
   
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
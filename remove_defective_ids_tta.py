# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os,sys
import shutil
import argparse
import numpy as np
import time

def main(args):
    cta_folder = args.cta
    tta_folder = args.tta
    file = args.file
    out_folder = args.out

    assert os.path.exists(cta_folder), f"CTA folder '{cta_folder}' does not exist"
    assert os.path.exists(tta_folder), f"TTA folder '{tta_folder}' does not exist"
    assert os.path.exists(file), f"File with IDs to be removed '{file}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent of output folder '{os.path.dirname(out_folder)}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Load IDs to be removed
    cids = np.loadtxt(file, dtype=str)
    for cid in cids:
        # Iterate through IDs
        # Access CTA and TTA files 
        cta_file = os.path.join(cta_folder, f"{cid}.nii.gz")
        tta_file = os.path.join(tta_folder, f"{cid}.nii.gz")
        tta_norm_file = os.path.join(tta_folder, f"{cid}_norm.nii.gz")
        tta_bin_file = os.path.join(tta_folder, f"{cid}_bin.nii.gz")
        tta_early_file = os.path.join(tta_folder, f"{cid}_early.nii.gz")
        tta_late_file = os.path.join(tta_folder, f"{cid}_late.nii.gz")

        files = [cta_file, tta_file, tta_norm_file, 
                tta_bin_file, tta_early_file, tta_late_file] 
        
        for enum_file, file_ in enumerate(files):
            if os.path.exists(file_):
                out = os.path.join(out_folder, "tta_discarded")
                if enum_file == 0:
                    out = os.path.join(out_folder, "cta_discarded")

                if not(os.path.exists(out)):
                    os.makedirs(out)

                outfile = os.path.join(out, os.path.basename(file_))
                shutil.move(file_, outfile)

def get_args():
    # Identify those case IDs from the registration process to be removed for further preprocessing 
    parser = argparse.ArgumentParser()
    parser.add_argument("--cta", help="CTA folder", required=True, type=str)
    parser.add_argument("--tta", help="TTA folder", required=True, type=str)
    parser.add_argument("--file", help="File with IDs to be removed", required=True, type=str)
    parser.add_argument("--out", help="Output folder", required=True, type=str)
   
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import numpy as np
import time
import argparse
import SimpleITK as sitk
from typing import Union
import shutil
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def process_id(data_file, data_folder, masked_folder, outfolder):
    cid = data_file.replace("_0000.nii.gz", "")
    outfile = os.path.join(outfolder, f"{cid}.png")
    if not (os.path.exists(outfile)):

        print(cid)
        full_file = os.path.join(data_folder, f"{cid}_0000.nii.gz")
        masked_file = os.path.join(masked_folder, f"{cid}_0000.nii.gz")

        data = sitk.GetArrayFromImage(sitk.ReadImage(full_file)).squeeze()
        masked = sitk.GetArrayFromImage(sitk.ReadImage(masked_file)).squeeze()

        plt.figure()
        plt.subplot(321)
        plt.imshow(data[data.shape[0] // 2], cmap="gray")
        plt.subplot(322)
        plt.imshow(masked[masked.shape[0] // 2], cmap="gray")
        plt.subplot(323)
        plt.imshow(data[:, data.shape[1] // 2], cmap="gray")
        plt.subplot(324)
        plt.imshow(masked[:, masked.shape[1] // 2], cmap="gray")
        plt.subplot(325)
        plt.imshow(data[:, :, data.shape[2] // 2], cmap="gray")
        plt.subplot(326)
        plt.imshow(masked[:, :, masked.shape[2] // 2], cmap="gray")
        plt.savefig(outfile)
        plt.close()


def main(args):
    data_folder = args.d
    masked_folder = args.m
    outfolder = args.o
    workers = args.np

    assert os.path.exists(data_folder), f"Data folder '{data_folder}' does not exist"
    assert os.path.exists(
        masked_folder
    ), f"Masked folder '{masked_folder}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Parent output folder '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    if not (os.path.exists(outfolder)):
        os.makedirs(outfolder)

    data_files = sorted(os.listdir(data_folder))

    Parallel(n_jobs=workers)(
        delayed(process_id)(data_file, data_folder, masked_folder, outfolder)
        for data_file in data_files
    )


def get_args():
    # Check if data for translation model and data for modelling is aligned
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="Input data folder", required=True, type=str)
    parser.add_argument("--m", help="Input masked folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument("--np", help="Parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)

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


def extreme_slices(segmentation: np.ndarray) -> Union[int, int]:
    """
    Derive extreme slices from brain segmentation

    Params
    ------
    segmentation : brain segmentation

    Returns
    -------
    start_slice : low axial limit
    high_lim : high axial limit

    """
    # Find the indices along the z-axis that have non-zero values
    non_empty_slices = np.any(segmentation, axis=(1, 2))

    # Get the first and last slice index where there is brain tissue
    start_slice = np.argmax(non_empty_slices)
    end_slice = len(non_empty_slices) - 1 - np.argmax(non_empty_slices[::-1])

    return start_slice, end_slice


def iterate(
    infolder: os.PathLike,
    infile: os.PathLike,
    brain_folder: os.PathLike,
    outfolder: os.PathLike,
):
    if infile.endswith(".nii.gz"):
        cid = infile.replace(".nii.gz", "")
        # cid = cid.replace("fast_", "")
        full_file = os.path.join(infolder, f"{cid}_0000.nii.gz")
        brain_file = os.path.join(brain_folder, f"{cid}.nii.gz")
        outfile = os.path.join(outfolder, f"{cid}_0000.nii.gz")
        if (
            os.path.exists(full_file)
            and os.path.exists(brain_file)
            and not (os.path.exists(outfile))
        ):
            image = sitk.ReadImage(full_file)
            img = sitk.GetArrayFromImage(image)
            brain = sitk.GetArrayFromImage(sitk.ReadImage(brain_file))
            outimg = img.copy()
            outimg[brain == 0] = -1024
            # extremes = extreme_slices(segmentation=brain)

            # outimg = outimg[extremes[0] : (extremes[-1] + 1)]
            outimage = sitk.GetImageFromArray(outimg)
            outimage.SetSpacing(image.GetSpacing())
            outimage.SetDirection(image.GetDirection())
            outimage.SetOrigin(image.GetOrigin())
            sitk.WriteImage(outimage, outfile)


def main(args):
    infolder = args.i
    brain_folder = args.b
    outfolder = args.o
    workers = args.np

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(
        brain_folder
    ), f"Brain segmentation folder '{brain_folder}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Parent output folder '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Zero or negative parallel workers"

    if not (os.path.exists(outfolder)):
        # Create output folder if it does not exist
        os.makedirs(outfolder)

    # Iterate through files
    infiles = sorted(os.listdir(infolder))
    Parallel(n_jobs=workers)(
        delayed(iterate)(infolder, infile, brain_folder, outfolder)
        for infile in infiles
        if ".nii.gz" in infile
    )


def get_args():
    # Prepare data for Amsterdam project translation model (brain X images)
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input folder", required=True, type=str)
    parser.add_argument("--b", help="Brain segm folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument("--np", help="Parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)

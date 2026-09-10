# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import SimpleITK as sitk
import numpy as np
import argparse
import time
from joblib import Parallel, delayed


def process_case(folder: os.PathLike, file: os.PathLike, outfolder: os.PathLike):
    outfile = os.path.join(outfolder, file)

    if not (os.path.exists(outfile)):
        full_file = os.path.join(folder, file)
        image = sitk.ReadImage(full_file)
        img = sitk.GetArrayFromImage(image)
        tta_bin = (img > 0).astype(np.uint8)
        tta_bin_image = sitk.GetImageFromArray(tta_bin)
        tta_bin_image.CopyInformation(image)
        sitk.WriteImage(tta_bin_image, outfile)


def main(args):
    infolder = args.i
    outfolder = args.o
    workers = args.np

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Output parent directory '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Zero or negative number of workers"

    if not (os.path.exists(outfolder)):
        os.makedirs(outfolder)

    files = sorted(os.listdir(infolder))

    Parallel(n_jobs=workers)(
        delayed(process_case)(infolder, file, outfolder)
        for file in files
        if file.endswith(".nii.gz")
    )


def get_args():
    parser = argparse.ArgumentParser(description="Binarize TTA maps")
    parser.add_argument(
        "--i", help="Folder with time information", required=True, type=str
    )
    parser.add_argument(
        "--o",
        help="Output folder with binarized time-vessel maps",
        required=True,
        type=str,
    )
    parser.add_argument("--np", help="Number of parallel workers", default=4, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)

# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import numpy as np
import argparse
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy.stats import rankdata
import time


def process_file(time_file, time_folder, gt_folder):
    cid = time_file.replace(".nii.gz", "")
    gt_file = os.path.join(gt_folder, f"{cid}_boxes_gt.npz")
    full_file = os.path.join(time_folder, time_file)

    tp_boxes = np.load(gt_file)["boxes"]

    if tp_boxes.shape[0] > 0:
        tta = sitk.GetArrayFromImage(sitk.ReadImage(full_file))
        # tta_vals = tta[tta > 0]
        # tta_vals = rankdata(tta_vals, method="average")
        # tta_vals = (tta_vals - 1) / (tta_vals.shape[0] - 1)
        # tta[tta > 0] = tta_vals

        # Iterate through TP boxes
        for i in range(tp_boxes.shape[0]):
            tp_box = tp_boxes[i].astype(int)

            patch = tta[
                tp_box[0] : tp_box[2], tp_box[1] : tp_box[3], tp_box[-2] : tp_box[-1]
            ]

            t_vals = patch[patch > 0]

            print(
                cid,
                t_vals.min(),
                t_vals.max(),
                t_vals.mean(),
                np.median(t_vals),
                np.percentile(t_vals, 5),
                np.percentile(t_vals, 25),
                np.percentile(t_vals, 75),
                np.percentile(t_vals, 95),
            )


def main(args):
    time_folder = args.t
    gt_folder = args.g
    workers = args.np

    assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    assert os.path.exists(
        gt_folder
    ), f"Ground-truth folder '{gt_folder}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Iterate through IDs
    time_files = sorted(os.listdir(time_folder))
    Parallel(n_jobs=workers)(
        delayed(process_file)(time_file, time_folder, gt_folder)
        for time_file in time_files
        if ".nii.gz" in time_file
    )


def get_args():
    # Inspect time values for true positives
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", help="Time folder", required=True, type=str)
    parser.add_argument("--g", help="Ground-truth folder", required=True, type=str)
    parser.add_argument("--np", help="Number of parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

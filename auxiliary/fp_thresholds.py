# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import numpy as np
from joblib import Parallel, delayed
import argparse
import time
from nndet.core.boxes.ops_np import box_iou_np
from nndet.io.load import load_pickle, load_json


def process_file(file, pred_folder, gt_folder, thr):
    cid = file.replace("_boxes.pkl", "")
    pred_file = os.path.join(pred_folder, file)
    gt_file = os.path.join(gt_folder, f"{cid}_boxes_gt.npz")

    pred = load_pickle(pred_file)
    gt = np.load(gt_file)["boxes"]

    pred_boxes, pred_scores = pred["pred_boxes"], pred["pred_scores"]

    ind = np.argwhere(pred_scores >= thr)
    pred_boxes = pred_boxes[ind]

    if gt.shape[0] == 0:
        num = pred_boxes.shape[0]
        print(cid, num)
        return num

    if pred_boxes.shape[0] > 0:
        pred_boxes = pred_boxes.squeeze()
        if len(pred_boxes.shape) == 1:
            pred_boxes = np.expand_dims(pred_boxes, 0)
        iou = box_iou_np(pred_boxes, gt)
        tp_inds = np.where(iou >= 0.10)[0]
        fp_inds = np.setdiff1d(np.arange(pred_boxes.shape[0]), tp_inds)
        print(cid, fp_inds.shape[0])
        return fp_inds.shape[0]

    print(cid, 0)
    return 0


def main(args):
    pred_folder = args.p
    gt_folder = args.g
    thr_file = args.t
    workers = args.np

    assert os.path.exists(
        pred_folder
    ), f"Prediction folder '{pred_folder}' does not exist"
    assert os.path.exists(
        gt_folder
    ), f"Ground-truth folder '{gt_folder}' does not exist"
    assert os.path.exists(thr_file) and thr_file.endswith(
        ".json"
    ), f"Threshold '{thr_file}' does not exist or is not .json"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Load threshold
    thr = load_json(thr_file)["thr_box"]

    # Get list of cases
    files = sorted(os.listdir(pred_folder))

    fps = Parallel(n_jobs=workers)(
        delayed(process_file)(file, pred_folder, gt_folder, thr)
        for file in files
        if file.endswith("_boxes.pkl")
    )

    print(sum(fps))


def get_args():
    # Measure number of false positives removed after thresholding
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", help="Prediction folder", required=True, type=str)
    parser.add_argument("--g", help="Ground-truth folder", required=True, type=str)
    parser.add_argument("--t", help="Threshold file", required=True, type=str)
    parser.add_argument("--np", help="Number of parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import argparse
import numpy as np
from joblib import Parallel, delayed
import time
from nndet.io.load import load_pickle
from nndet.core.boxes.ops_np import box_iou_np


def process_case(file, ref_folder, model_folder, gt_folder):
    if file.endswith(".pkl"):
        cid = file.replace("_boxes.pkl", "")

        ref_file = os.path.join(ref_folder, file)
        model_file = os.path.join(model_folder, f"{cid}_boxes.pkl")
        gt_file = os.path.join(gt_folder, f"{cid}_boxes_gt.npz")

        if (
            os.path.exists(ref_file)
            and os.path.exists(model_file)
            and os.path.exists(gt_file)
        ):
            ref_pred = load_pickle(ref_file)["pred_boxes"]
            ref_scores = load_pickle(ref_file)["pred_scores"]
            model_pred = load_pickle(model_file)["pred_boxes"]
            model_scores = load_pickle(model_file)["pred_scores"]
            gt = np.load(gt_file)["boxes"]

            thr = 0.375
            ind_ref = np.where(ref_scores >= thr)[0]
            ind_model = np.where(model_scores >= thr)[0]
            ref_pred = ref_pred[ind_ref]
            model_pred = model_pred[ind_model]

            print(cid, ref_pred.shape, model_pred.shape, gt.shape)

            if gt.shape[0] > 0:
                iou_ref = box_iou_np(gt, ref_pred)
                tp_inds = np.where(iou_ref >= 0.10)[0]
                if tp_inds.shape[0] > 0:
                    iou_model = box_iou_np(gt, model_pred)
                    tp_inds_model = np.where(iou_model >= 0.10)[0]

                    if tp_inds_model.shape[0] < tp_inds.shape[0]:
                        print(
                            cid, tp_inds.shape[0], tp_inds_model.shape[0], gt.shape[0]
                        )


def main(args):
    ref_folder = args.ref
    model_folder = args.m
    gt_folder = args.gt
    workers = args.np

    assert os.path.exists(ref_folder), f"Reference folder '{ref_folder}' does not exist"
    assert os.path.exists(model_folder), f"Model folder '{model_folder}' does not exist"
    assert os.path.exists(
        gt_folder
    ), f"Ground-truth folder '{gt_folder}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    ref_files = sorted(os.listdir(ref_folder))
    Parallel(n_jobs=workers)(
        delayed(process_case)(file, ref_folder, model_folder, gt_folder)
        for file in ref_files
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Get case IDs where TPs are completely removed"
    )
    parser.add_argument("--ref", help="Folder for reference predictions", type=str)
    parser.add_argument("--m", help="Folder for method's predictions", type=str)
    parser.add_argument("--gt", help="Ground truth folder", type=str)
    parser.add_argument("--np", help="Number of parallel workers", type=int, default=4)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

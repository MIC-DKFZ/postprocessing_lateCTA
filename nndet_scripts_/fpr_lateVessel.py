# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os,sys
import numpy as np
import argparse
import time
import SimpleITK as sitk
import shutil
from typing import Union

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import load_data, write_data

def filter_out_fps(boxes : np.ndarray, scores : np.ndarray, labels : np.ndarray, keep : np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out false positives

    Params
    ------
    boxes : input boxes
    scores : input scores
    labels : input labels
    keep : flag telling whether to keep or not predicted box

    Returns
    -------
    out_boxes : filtered boxes
    out_scores : filtered scores
    out_labels : filtered labels
    
    """
    # Obtain keep ratio 
    ratio = keep.sum() / (boxes.shape[0] + np.finfo(float).eps)

    if ratio < 0.1:
        # If less than 10% of boxes are remaining, we may be removing true positives
        # Remove only half of the boxes to be removed, with the lowest scores
        ind_remove = np.where(keep == False)[0]
        scores_remove = scores[ind_remove]       
        scores_argsort = np.argsort(scores_remove)
        ind_scores_remove = scores_argsort[:scores_argsort.shape[0]//2]
        ind_remove_final = ind_remove[ind_scores_remove]
        keep = np.ones(keep.shape[0], dtype=bool)
        keep[ind_remove_final] = False

    out_boxes, out_scores, out_labels = boxes[keep] , scores[keep], labels[keep]

    return out_boxes, out_scores, out_labels


def main(args):
    pred_folder = args.pred
    segm_folder = args.segm
    # time_folder = args.time
    out_folder = args.out

    assert os.path.exists(pred_folder), f"Prediction folder '{pred_folder}' does not exist"
    assert os.path.exists(segm_folder), f"Segmentation folder '{segm_folder}' does not exist"
    # assert os.path.exists(time_folder), f"Time folder '{time_folder}' does not exist"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Iterate through prediction files
    files = sorted(os.listdir(pred_folder))
    for file in files:
        if ".pkl" in file:
            # Load prediction information 
            cid = file.replace("_boxes.pkl", "")
            pred_file = os.path.join(pred_folder, file)
            pred = load_data(pred_file)
            out_dict = pred.copy()
            boxes, scores, labels = pred["pred_boxes"], pred["pred_scores"], pred["pred_labels"]

            outfile = os.path.join(out_folder, file)

            # Load segmentation information
            segm_file =  os.path.join(segm_folder, f"{cid}.nii.gz")
            segm = sitk.GetArrayFromImage(sitk.ReadImage(segm_file))
            segm = (segm > 0).astype(np.float32)

            # Load time information
            # time_file =  os.path.join(time_folder, f"{cid}.nii.gz")
            # time_segm = sitk.GetArrayFromImage(sitk.ReadImage(time_file))
            # time_segm = (time_segm > 0).astype(np.float32)

            # Filter late area segmentation with vessel segmentation
            #  segm *= time_segm 
            
            # Iterate through boxes
            keep = []
            if boxes.shape[0] > 0:   
                for i in range(boxes.shape[0]):
                    box = boxes[i].astype(int)
                    patch = segm[box[0]:box[2],box[1]:box[3],box[-2]:box[-1]] 
                    k = False
                    if patch.sum() > 0:
                        k = True
                    keep.append(k)
            else:
                shutil.copyfile(pred_file, outfile)
                continue

            # Save filtered information 
            keep = np.array(keep, dtype=bool)

            boxes_filtered, scores_filtered, labels_filtered = filter_out_fps(boxes = boxes, scores=scores, labels=labels, keep=keep) 
            # boxes_filtered, scores_filtered, labels_filtered = boxes[keep] , scores[keep], labels[keep]
            
            print(cid, boxes.shape[0] , boxes_filtered.shape[0])
            out_dict["pred_boxes"] = boxes_filtered 
            out_dict["pred_labels"] = labels_filtered 
            out_dict["pred_scores"] = scores_filtered 

            
            write_data(data = out_dict, filename=outfile)


def get_args():
    # Remove predictions outside of lately enhanced regions 
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="Prediction folder", type=str)
    parser.add_argument("--segm", help="Late vessel segmentation", type=str)
    # parser.add_argument("--time", help="Time image segmentation", type=str)
    parser.add_argument("--out", help="Output folder with FPR", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
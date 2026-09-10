# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os,sys
import numpy as np
import argparse
import SimpleITK as sitk
import time
from joblib import Parallel, delayed
from scipy.ndimage import label
from typing import Union


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from utils.load_save import load_data, write_data


def plot_differences(boxes1: np.ndarray, boxes2: np.ndarray, shape : tuple) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot differences between predicted boxes

    Params
    ------
    boxes1 : predicted boxes 1
    boxes2 : predicted boxes 2
    shape : image shape


    Returns
    -------
    mask : mask with differences (1: removed boxes, 2: preserved boxes)
    preserved_boxes : preserved boxes
    removed_boxes : removed boxes
    
    """
    # Compare sets of boxes
    # View each row as a single element of a compound dtype
    boxes1_view = boxes1.copy().view([('', boxes1.dtype)] * boxes1.shape[1])
    boxes2_view = boxes2.copy().view([('', boxes2.dtype)] * boxes2.shape[1])

    # Find common rows using intersect1d
    common_rows, ind_common, _ = np.intersect1d(boxes1_view, 
                                                boxes2_view, 
                                                return_indices=True) 
    ind_different = np.setdiff1d(np.arange(boxes1.shape[0]),
                                 ind_common)
    ind_common = ind_common.tolist()

    # Set to ones those boxes common to both sets and to twos non-overlapping boxes
    mask = np.zeros(shape, dtype=np.uint8)
    for i,box in enumerate(boxes1):
        box_int = box.astype(int)
        if i in ind_different:
            mask[box_int[0]: box_int[2], 
                 box_int[1] : box_int[3], 
                 box_int[-2]: box_int[-1]] = 1
        else:
            mask[box_int[0]: box_int[2], 
                 box_int[1] : box_int[3], 
                 box_int[-2]: box_int[-1]] = 2
    
    preserved_boxes = boxes1[ind_common]
    removed_boxes = boxes1[ind_different]

    return mask, preserved_boxes, removed_boxes

    
def obtain_stats(boxes : np.ndarray, time_seg : np.ndarray, tag : str = "") -> dict:
    """
    Derive statistics on set of boxes (time and vessel wise)

    Params
    ------
    boxes : boxes to be analyzed
    time_seg : time vessel segmentation
    tag : statistics tag ("preserved" or "removed")

    Returns
    -------
    stats : boxes' statistics on time and vesse-wise positiion
    
    """

    # Store relevant information 
    in_vessel, time_info = 0,[]    

    for box in boxes:
        box_int = box.astype(int)
        patch_time = time_seg[box_int[0]: box_int[2], 
                            box_int[1] : box_int[3], 
                            box_int[-2]: box_int[-1]] 
        
        patch_seg = time_seg[box_int[0]: box_int[2], 
                            box_int[1] : box_int[3], 
                            box_int[-2]: box_int[-1]] 
        
        if patch_seg.sum() > 0:
            in_vessel += 1
            time_val = np.median(patch_time[patch_seg > 0].flatten()) 
            time_info.append(time_val)

    # Derive final statistics
    in_vessel_fraction = in_vessel*100/(boxes.shape[0] + np.finfo(float).eps)
    stats = {f"in_vessel_{tag}" : in_vessel_fraction} 

    if len(time_info) > 0:
        # There is time information available
        time_info = np.array(time_info)
        stats[f"time_mean_{tag}"] = float(time_info.mean())
        stats[f"time_std_{tag}"] = float(time_info.std())
        stats[f"time_max_{tag}"] = float(np.percentile(time_info, 99.5))
        stats[f"time_min_{tag}"] = float(np.percentile(time_info, 0.5))
    else:
        stats[f"time_mean_{tag}"] = 0.0
        stats[f"time_std_{tag}"] = 0.0
        stats[f"time_max_{tag}"] = 0.0
        stats[f"time_min_{tag}"] = 0.0

    return stats


def analyze_tps(gt_folder : os.PathLike, mask : np.ndarray, time_seg : np.ndarray, cid : str):
    """
    Analyze true positives in set. See if the true positives are maintained,
    if the true positives lie close to vessels, and the connected component
    size containing the true positive information

    Params
    ------
    gt_folder: folder with true positive information
    mask : volume with preserved and discarded box information
    time_seg : time map information
    cid : case ID
    
    Returns
    -------
    tp_stats : true positive statistics
    
    """
    # True positive file information 
    tp_file = os.path.join(gt_folder, f"{cid}_boxes_gt.npz")
    tp_boxes = np.load(tp_file)["boxes"] 

    # Obtain connected component image
    label_img, _ = label((time_seg > 0).astype(np.uint8))

    # Derive connected component sizes
    labels, sizes = np.unique(label_img.flatten(), 
                              return_counts=True)
    sorted_sizes = np.sort(sizes)

    # See if the true positive information is preserved 
    tp_preserved, in_vessel, rel_size, time_mean = 1, 0, 0, 0
    if tp_boxes.shape[0] > 0:
        for tp_box in tp_boxes:
            tp_box_int = tp_box.astype(int)
            tp_patch = mask[tp_box_int[0] : tp_box_int[2],
                            tp_box_int[1] : tp_box_int[3],
                            tp_box_int[-2] : tp_box_int[-1]] 
            time_patch = time_seg[tp_box_int[0] : tp_box_int[2],
                            tp_box_int[1] : tp_box_int[3],
                            tp_box_int[-2] : tp_box_int[-1]]             
            label_patch = label_img[tp_box_int[0] : tp_box_int[2],
                            tp_box_int[1] : tp_box_int[3],
                            tp_box_int[-2] : tp_box_int[-1]] 
            
            if (time_patch > 0).any():
                in_vessel = 1
                time_vals = time_patch[time_patch > 0]
                time_mean = np.median(time_vals) 
                components = np.unique(label_patch.flatten())
                components = components[1:] # Discard background 

                # Get sizes in complete image of the components found
                _,ind_found,_ = np.intersect1d(labels, components, return_indices=True)
                if ind_found.shape[0] > 0: 
                    sizes_found = sizes[ind_found]
                    max_size = sizes_found.max()

                    # Get relative size of the biggest component found
                    size_ind = np.where(sorted_sizes == max_size)[0] 
                    rel_size = size_ind/(sorted_sizes.shape[0] + np.finfo(float).eps)
                    rel_size = rel_size[0] 
                   
            if np.array_equal(np.unique(tp_patch), np.array([0,1])):
                tp_preserved = 0
                break

    tp_stats = {"tp_preserved" : float(tp_preserved),
                "tp_cc_size" : float(rel_size),
                "tp_in_vessel" : float(in_vessel),
                "tp_time" : float(time_mean)}

    return tp_stats 


def process_case(file, t, f1, f2, out, ref): 
    # Load case ID  
    cid = file.replace("_boxes.pkl", "")

    # Load time map 
    time_file = os.path.join(t, f"{cid}_norm.nii.gz")
    assert os.path.exists(time_file), f"Time file '{time_file}' does not exist"
    time_image = sitk.ReadImage(time_file)
    time_seg = sitk.GetArrayFromImage(time_image)

    # Obtain preserved boxes and removed boxes
    file1 = os.path.join(f1, f"{cid}_boxes.pkl")
    file2 = os.path.join(f2, f"{cid}_boxes.pkl")
    assert os.path.exists(file1), f"Boxes file 1 '{file1}' does not exist"
    assert os.path.exists(file2), f"Boxes file 2 '{file2}' does not exist"

    boxes1 = load_data(filename=file1)["pred_boxes"] 
    boxes2 = load_data(filename=file2)["pred_boxes"]
    diff, preserved, removed = plot_differences(boxes1=boxes1,
                                                boxes2=boxes2,
                                                shape=time_seg.shape)
        
    diff_image = sitk.GetImageFromArray(diff.astype(np.uint8))
    diff_image.CopyInformation(time_image)
    outfile = os.path.join(out, f"{cid}.nii.gz")
    sitk.WriteImage(diff_image, outfile)

    # Obtain prediction statistics
    # Preserved boxes
    preserved_stats = obtain_stats(boxes=preserved,
                                    time_seg=time_seg, 
                                    tag="preserved")
    
    # Removed boxes
    removed_stats = obtain_stats(boxes=removed,
                                    time_seg=time_seg, 
                                    tag="removed")
    
    # Obtain ground-truth statistics
    gt_stats = analyze_tps(gt_folder=ref, 
                            mask=diff,
                            time_seg=time_seg,
                            cid = cid)
    

    # Combine statistics in an only dictionary
    preserved_stats.update(removed_stats)
    preserved_stats.update(gt_stats)
    outfile = os.path.join(out, f"{cid}.json")
    write_data(data=preserved_stats, filename=outfile)
    print(cid, preserved_stats) 

    return preserved_stats



def main(args):
    f1 = args.a
    f2 = args.b
    t = args.t
    ref = args.ref
    out = args.out

    assert os.path.exists(f1), f"FPR folder a) '{f1}' does not exist"
    assert os.path.exists(f2), f"FPR folder b) '{f2}' does not exist"
    assert os.path.exists(t), f"Time segmentation folder '{t}' does not exist"
    assert os.path.exists(ref), f"Reference folder '{ref}' does not exist"
    assert os.path.exists(os.path.dirname(out)), f"Parent output folder '{os.path.dirname(out)}' does not exist"

    if not(os.path.exists(out)):
        # Check if output folder exists
        # If it does not exist, create it  
        os.makedirs(out)

    # Iterate through predictions
    files = sorted(os.listdir(f1))
    preserved_tp, preserved_in_vessel, preserved_time_mean = [],[],[]
    removed_in_vessel, removed_time_mean, tp_cc_size, tp_time = [],[], [], []  

    combined_dicts = Parallel(n_jobs=6)(delayed(process_case)(file, t, f1, f2, out, ref) for file in files if ".pkl" in file)

    for combined in combined_dicts:        
        # Aggregate combined statistics
        preserved_tp.append(combined["tp_preserved"])
        preserved_in_vessel.append(combined["in_vessel_preserved"]) 
        preserved_time_mean.append(combined["time_mean_preserved"])
        removed_in_vessel.append(combined["in_vessel_removed"]) 
        removed_time_mean.append(combined["time_mean_removed"]) 
        tp_cc_size.append(combined["tp_cc_size"])
        tp_time.append(combined["tp_time"] )

    # Missed TPs
    print(f"Missed TPs: {len(preserved_tp)-sum(preserved_tp)}/{len(preserved_tp)}")

    # In vessel preserved rate
    print(f"Average preserved in vessel rate: {sum(preserved_in_vessel)/len(preserved_in_vessel)}%") 
    # Time preserved rate
    print(f"Average time in preserved boxes: {sum(preserved_time_mean)/len(preserved_time_mean)}") 

    # In vessel removed rate
    print(f"Average removed in vessel rate: {sum(removed_in_vessel)/len(removed_in_vessel)}%") 
    # Time removed rate
    print(f"Average time in removed boxes: {sum(removed_time_mean)/len(removed_time_mean)}")

    # TP connected component rank
    print(f"TP connected component rank : {sum(tp_cc_size) / len(tp_cc_size)}") 

    # TP time
    print(f"TP time: {sum(tp_time) / len(tp_time)}") 


def get_args():
    # Remove predictions outside of lately enhanced regions 
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", help="Reference prediction folder", type=str)
    parser.add_argument("--b", help="Comparison prediction folder", type=str)
    parser.add_argument("--t", help="Time vessel folder", type=str)
    parser.add_argument("--ref", help="Ground-truth folder", type=str)
    parser.add_argument("--out", help="Output folder with QA", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
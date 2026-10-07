import os,sys
import numpy as np
import SimpleITK as sitk
import time
import argparse
from skimage.morphology import skeletonize
from joblib import Parallel, delayed
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import load_json
import matplotlib.pyplot as plt

def dice_coefficient(mask1, mask2):
    """Dice = 2|A∩B| / (|A| + |B|)"""
    intersection = np.logical_and(mask1, mask2).sum()
    return 2.0 * intersection / (mask1.sum() + mask2.sum() + 1e-8)

def threshold(pred : np.ndarray, gt : np.ndarray, thr : float, mode : str = "upper"):
    """
    Compute segmentation performance when thresholding 
    prediction map, based on an only or several threshold values

    Params
    ------
    pred : prediction map
    gt : ground-truth map
    thr : thresholding value
    mode : thresholding mode ("lower" or "upper", default "upper")
    
    Returns
    -------
    metric : segmentation metric
    
    """
    if isinstance(pred, np.ndarray):
        pred = [pred]

    if isinstance(thr, float) or isinstance(thr, int):
        thr = [thr]

    if isinstance(mode, str):
        mode = [mode]

    # Derive segmentation structure
    maps = []
    for p,t,m in zip(pred, thr, mode):
        if m == "upper":
            maps.append(p > t)
        if m == "lower":
            maps.append(p < t)
    
    if len(maps) == 1:
        pred_map = maps[0]
    elif len(maps) > 1:
        s = np.zeros(maps[0].shape)
        for m in maps:
            s += m
        pred_map = (s > 0).astype(np.uint8)

    # Compute segmentation metric: Dice
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(pred_map[pred_map.shape[0]//2], cmap="gray")
    plt.subplot(122)
    plt.imshow(gt[pred_map.shape[0]//2], cmap="gray")
    plt.show()
    plt.close()
    """
    metric = 0.0
    if pred_map.sum() > 0:
        metric = dice_coefficient(mask1=pred_map.astype(bool), mask2=gt.astype(bool))

    print(thr, metric)

    return metric


def process_id(pred_folder : os.PathLike, file : os.PathLike, ref_folder : os.PathLike, brain_folder : os.PathLike, cfg : dict, workers : int = 6) -> dict:
    """
    Process ID

    Params
    ------
    pred_folder : prediction folder
    file : prediction file
    ref_folder : ground truth folder
    brain_folder : brain segmentation folder
    cfg : configuration
    workers : number of parallel workers

    Returns
    -------
    metrics = ID metrics obtained with different distance and logit thresholds

    """
    # Obtain case Id
    cid = file.replace(".npy", "")
    print(cid)

    # Load predicted and reference images, and brain segmentations
    pred = np.load(os.path.join(pred_folder, f"{cid}.npy"))
    pred = np.moveaxis(pred, -1, 0)
    pred = np.flip(pred,(2,3)) # TODO: flip predicted volumes as well!!
    gt_file = os.path.join(ref_folder, f"{cid}.nii.gz")
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
    brain_file = os.path.join(brain_folder, f"{cid}.nii.gz")
    brain = sitk.GetArrayFromImage(sitk.ReadImage(brain_file))
    tta_bin = (gt[-1] > 0).astype(np.uint8)

    # Distance intervals
    dist_int = np.arange(start = 0.0, stop=2.0, step=0.1)   

    # Logit intervals
    logit_int = np.arange(start=3.0, stop=40.0, step=1.0)

    dist_map = pred[0]*cfg["std"] # Remove normalization from distance map
    logit_map = pred[-1]*brain

    # Iterate over distances
    dist_vals = Parallel(n_jobs=workers)(delayed(threshold)(pred=dist_map,gt=tta_bin, thr=d, mode='lower') for d in dist_int)
    logit_vals = Parallel(n_jobs=workers)(delayed(threshold)(pred=logit_map,gt=tta_bin, thr=l, mode='upper') for l in logit_int)

    # Get pairs of distances and logits
    pairs, pair_names = [], []
    for d in dist_int:
        for l in logit_int:
            pairs.append([d,l])
            pair_names.append(f"dist_{d}_logit_{l}")

    dist_logit_vals = Parallel(n_jobs=workers)(delayed(threshold)(pred=[dist_map, logit_map],
                                                        gt=tta_bin, thr=p, 
                                                        mode=['lower','upper']) for p in pairs)
    
    out_dict = {} # Dictionary with metrics measured
    for i,d in zip(dist_int,dist_vals):
        out_dict[f"dist_{i}"] = d
    for i,l in zip(logit_int,logit_vals):
        out_dict[f"dist_{i}"] = l
    for p,d in zip(pair_names,dist_logit_vals):
        out_dict[p] = d

    return out_dict


def main(args):
    pred_folder = args.p
    ref_folder = args.r
    brain_folder = args.b
    cfg_file = args.c
    outfile = args.o
    workers = args.np

    assert os.path.exists(pred_folder), f"Prediction folder '{pred_folder}' does not exist"
    assert os.path.exists(brain_folder), f"Brain folder '{brain_folder}' does not exist"
    assert os.path.exists(ref_folder), f"Reference folder '{ref_folder}' does not exist"
    assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)), f"Parent output directory '{os.path.dirname(outfile)}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Load configuration for distance map normalization
    cfg = load_json(cfg_file)

    # Iterate through IDs
    files = sorted(os.listdir(pred_folder))
    out_dict = {}
    for file in files:
        if ".npy" in file:
            cid = file.replace(".npy", "")
            out_dict[cid] = process_id(pred_folder=pred_folder,
                                       file=file,
                                       ref_folder=ref_folder,
                                       brain_folder=brain_folder,
                                       cfg=cfg, 
                                       workers=workers)
    
    # Store everything in a dataframe
    df = pd.DataFrame.from_dict(out_dict, orient='index')
    df.to_csv(outfile)
    

def get_args():
    # Remove predictions outside of lately enhanced regions 
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", help="Prediction folder", required=True, type=str)
    parser.add_argument("--c", help="Configuration file", required=True, type=str)
    parser.add_argument("--r", help="Reference folder", required=True, type=str)
    parser.add_argument("--b", help="Brain folder", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    parser.add_argument("--np", help="Workers", default=6, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
import os, sys
import numpy as np
from nndet.io.load import load_pickle, save_pickle
import SimpleITK as sitk
import argparse
from joblib import Parallel, delayed
from typing import Union
import time


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


def return_valid_inds(boxes: np.ndarray, extremes: list) -> np.ndarray:
    """
    Return valid indexes to only keep boxes in brain areas

    Params
    ------
    boxes : input boxes
    extremes : brain slice extremes

    Returns
    -------
    valid_inds : valid indices

    """
    x0, xf = boxes[:, 0], boxes[:, 2]
    valid_inds = np.where((x0 >= extremes[0]) & (xf < extremes[-1]))[0]

    return valid_inds


def process_id(cid: str, data_folder: os.PathLike, pred_folder: os.PathLike):
    data_file = os.path.join(data_folder, f"{cid}.nii.gz")
    pred_file = os.path.join(pred_folder, f"{cid}_boxes.pkl")
    outfile = os.path.join(pred_folder, f"{cid}_boxes_new.pkl")

    # Load brain data file
    segm = sitk.GetArrayFromImage(sitk.ReadImage(data_file))
    extremes = extreme_slices(segmentation=segm)

    # Prediction loading
    pred = load_pickle(pred_file)
    final_pred = pred.copy()
    pred_boxes, pred_scores, pred_labels = (
        pred["pred_boxes"],
        pred["pred_scores"],
        pred["pred_labels"],
    )
    valid_inds = return_valid_inds(boxes=pred_boxes, extremes=extremes)
    final_pred["pred_boxes"] = pred_boxes[valid_inds]
    final_pred["pred_scores"] = pred_scores[valid_inds]
    final_pred["pred_labels"] = pred_labels[valid_inds]

    save_pickle(data=final_pred, path=outfile)


def main(args):
    data_folder = args.d
    file_cases = args.i
    pred_folder = args.p
    workers = args.np

    assert os.path.exists(data_folder), f"Data folder '{data_folder}' does not exist"
    assert os.path.exists(
        file_cases
    ), f"File with case information '{file_cases}' does not exist"
    assert os.path.exists(
        pred_folder
    ), f"Prediction folder '{pred_folder}' does not exist"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Load IDs
    cids = np.loadtxt(file_cases, dtype=str)

    Parallel(n_jobs=workers)(
        delayed(process_id)(cid=cid, data_folder=data_folder, pred_folder=pred_folder)
        for cid in cids
    )


def get_args():
    # Remove box predictions outside brain from specific cases
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="Input data folder", required=True, type=str)
    parser.add_argument("--i", help="Case IDs file", required=True, type=str)
    parser.add_argument("--p", help="Prediction folder", required=True, type=str)
    parser.add_argument("--np", help="Parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print("Time ellapsed (seconds): ", time.time() - t1)

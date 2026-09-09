import os,sys
import argparse
import numpy as np
import time
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import load_data


def retrieve_task_folder(key : str, task_id : str) -> str:
    """
    Obtain task folder from key directory

    Params
    ------
    key : key for directory
    task_id : ID for task of interest

    Returns
    -------
    task_folder : task folder
    
    """
    assert os.getenv(key) is not None, f"Environment key '{key}' not setup"
    folders = np.array(sorted(os.listdir(os.getenv(key))), dtype=str)
    mask = np.char.startswith(folders, f"Task{task_id}")

    assert mask.sum() > 0, f"Task folder with ID '{task_id}' not found, or duplicated" 
    task_folder = folders[mask][0]  

    return task_folder

def box_area_3d_np(
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Notes:
        always prefer using the n-D version since it takes care of data types.

    See Also:
        `nndet.core.boxes.ops.box_area_3d`
    """
    return (
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1])
        * (boxes[:, 5] - boxes[:, 4])
    )

def box_iou_3d_np(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]

    Returns:
        ndarray: the NxM iou matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    area1 = box_area_3d_np(boxes1)
    area2 = box_area_3d_np(boxes2)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = np.maximum(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = np.minimum(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    inter = (
        np.clip((x2 - x1), a_min=0, a_max=None)
        * np.clip((y2 - y1), a_min=0, a_max=None)
        * np.clip((z2 - z1), a_min=0, a_max=None)
    )  # [N, M]
    return inter / (area1[:, None] + area2 - inter)


def main(args):
    task_id = args.t
    model_id = args.m
    fold = args.f
    out = args.o
    thr = args.thr

    assert os.path.exists(os.path.dirname(out)), f"Parent folder of output file '{out}' does not exist"

    # Set up data and model task folder
    task_folder_data = retrieve_task_folder(key = "det_data", task_id=task_id)
    task_folder_model = retrieve_task_folder(key = "det_models", task_id=task_id)

    # Set up result folder
    fold_id = "consolidated" if fold == -1 else f"fold{fold}"
    result_folder = os.path.join(os.getenv("det_models"), task_folder_model, model_id, fold_id, "val_predictions")
    assert os.path.exists(result_folder), f"Result folder '{result_folder}' does not exist"

    # Set up ground-truth folder  
    gt_folder = os.path.join(os.getenv("det_data"), task_folder_data, "preprocessed", "labelsTr")
    assert os.path.exists(gt_folder), f"Ground-truth folder '{gt_folder}' does not exist"

    # Iterate through cases
    files = sorted(os.listdir(result_folder)) 
    out_dict = {"cid" : [], "hit" : [], "hit_score" : [], "max_fp": [], "tps" : [], "fps" : [], "tp_mean" : [], "fp_mean" : []} 
    for file in files:
        if ".pkl" in file:
            cid = file.replace("_boxes.pkl", "")
            # Retrieve predicted and ground-truth files, and read them 
            pred_file = os.path.join(result_folder, file)
            gt_file = os.path.join(gt_folder, f"{cid}_boxes_gt.npz")

            pred = load_data(filename=pred_file)
            pred_boxes, pred_scores = pred["pred_boxes"], pred["pred_scores"]  
            gt_data = np.load(gt_file)
            gt_boxes = gt_data["boxes"]
            iou = box_iou_3d_np(boxes1 = gt_boxes, boxes2=pred_boxes)
            iou_mask = np.sum(iou >= thr, axis=0)
            tp_mask, fp_mask = iou_mask > 0, iou_mask == 0

            # Compute number of TPs and FPs
            tps, fps = tp_mask.sum(), fp_mask.sum() 

            # Determine if there has been a hit 
            hit = tps > 0
            hit_score = pred_scores[tp_mask].max() if hit else 0

            # Determine maximum score FP
            max_score_fp = pred_scores[fp_mask].max() if fps > 0 else 0

            # Determine mean TP and FP 
            mean_tp = pred_scores[tp_mask].mean() if tp_mask.shape[-1] > 0 else 0 
            mean_fp = pred_scores[fp_mask].mean() if fp_mask.shape[-1] > 0 else 0 

            # Update output dictionary 
            out_dict["cid"].append(cid)
            out_dict["fp_mean"].append(mean_fp) 
            out_dict["tp_mean"].append(mean_tp) 
            out_dict["hit"].append(float(hit))
            out_dict["hit_score"].append(hit_score)
            out_dict["max_fp"].append(max_score_fp)
            out_dict["tps"].append(tps)
            out_dict["fps"].append(fps)    

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(out)

    

def get_args():
    # Analyze thrombus detection results
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", help="Task", required=True, type=str)
    parser.add_argument("--m", help="Model", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    parser.add_argument("--f", help="Fold", default=0, type=int)
    parser.add_argument("--thr", help="Threshold", default=0.1, type=float)
    
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(time.time()-t1)
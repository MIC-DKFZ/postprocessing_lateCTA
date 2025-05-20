import os,sys
import numpy as np
import argparse
import time
import SimpleITK as sitk
import shutil
from typing import Union
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import load_data, write_data
from derive_late_image import extract_late_vessels



def get_k_closest_points(binary_mask : np.ndarray, query_point : np.ndarray, k : int) -> Union[np.ndarray, np.ndarray]:
    """
    Obtain the k closest points from a point to a
    binary mask

    Params
    ------
    binary_mask : binary mask
    query_point : input point of interest
    k : number of points to retrieve

    Returns
    -------
    points : k closest points 
    dists : respective distances
    
    """
    # Step 1: Get all foreground points (value == 1)
    foreground_coords = np.argwhere(binary_mask == 1)
    
    # Step 2: Build KDTree for efficient nearest neighbor search
    tree = cKDTree(foreground_coords)
    
    # Step 3: Query the k closest points
    dists, indices = tree.query(query_point, k=k)
    
    # Step 4: Return the k closest coordinates and their distances
    return foreground_coords[indices], dists



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


def box_acceptance(box : np.ndarray, score : float, dist : np.ndarray, time_map : np.ndarray, collateral : np.ndarray, k : int) -> float:
    """
    Accept or reject box based on score, position in time map
    and distance to vessel of interest

    Penalize boxes falling very close to excessively late vessels

    Params
    ------
    box : input box
    score : input score
    dist : distance map
    time_map : time map
    collateral : segmentation with collateral flow vessels
    k : return k closest points from vasculature in boxes outside the vasculature


    Returns
    -------
    out_score : resulting score from analysis
    
    """
    # Integer coordinate box 
    box_int = box.astype(int)

    # Analyze box position in distance map and in time map
    dist_vals = dist[box_int[0]:box_int[2],
                     box_int[1]:box_int[3],
                     box_int[-2]:box_int[-1]]
    
    # Derive collateral flow area
    collateral_roi = collateral[box_int[0]:box_int[2],
                        box_int[1]:box_int[3],
                        box_int[-2]:box_int[-1]]
    
    if collateral_roi.sum() > 0:
        # Box lies in a collateral flow area, keep the same score 
        return score

    if (dist_vals <= 0).any():
        # The box lies on the vasculature
        vessel_coords = np.where(dist_vals.flatten() <= 0)[0]
        time_vals = time_map[box_int[0]:box_int[2],
                        box_int[1]:box_int[3],
                        box_int[-2]:box_int[-1]].flatten()
        t_val = np.median(time_vals[vessel_coords].flatten())
    else:
        # box_center = np.array([box[0]+box[2], 
        #                        box[1]+box[3], 
        #                        box[-2]+box[-1]])//2
        # binary_map = (dist <= 0).astype(int)

        # coords, dists = get_k_closest_points(binary_mask=binary_map, query_point=box_center, k = k)
        # inv_dists = 1 / (dists + np.finfo(float).eps)
        # weights = inv_dists / (inv_dists.sum() + np.finfo(float).eps)

        # Retrieve time values from the k closest coordinates, 
        # weighting them with the inverse of the distances
        # t_val = np.dot(weights, time_map[coords[:,0], coords[:,1], coords[:,2]])
        
        # Box outside vasculature, remove it  
        out_score = 0.0
        return out_score


    # Downweigh score based on retrieved time value
    # collateral_times = time_map[collateral > 0] 

    # Set up a maximum normalization time, 
    # the minimum time found in collateral flow area
    # t_max = collateral_times.min()  
    t_max = time_map.max()

    if t_val > t_max:
        # Boxes close to collateral vessels could still be needed to have their original score
        return score

    out_score = score*(1-(t_val/(t_max + np.finfo(float).eps))) 

    # print(score, out_score, t_val)

    return out_score
         



def main(args):
    pred_folder = args.pred
    segm_folder = args.segm
    dist_folder = args.dist
    out_folder = args.out
    k = args.k

    assert os.path.exists(pred_folder), f"Prediction folder '{pred_folder}' does not exist"
    assert os.path.exists(segm_folder), f"Segmentation folder '{segm_folder}' does not exist"
    assert os.path.exists(dist_folder), f"Distance map folder '{dist_folder}' does not exist"
    assert (k > 0), "k is zero or negative"
    assert os.path.exists(os.path.dirname(out_folder)), f"Parent output folder '{out_folder}' does not exist"

    if not(os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Iterate through prediction files
    files = sorted(os.listdir(pred_folder))
    for file in files:
        if ".pkl" in file:
            t1 = time.time()
            # Load prediction information 
            cid = file.replace("_boxes.pkl", "")
            print(cid)
            pred_file = os.path.join(pred_folder, file)
            pred = load_data(pred_file)
            out_dict = pred.copy()
            boxes, scores, labels = pred["pred_boxes"], pred["pred_scores"], pred["pred_labels"]

            outfile = os.path.join(out_folder, file)

            # Load time information
            segm_file =  os.path.join(segm_folder, f"{cid}_norm.nii.gz")
            time_map = sitk.GetArrayFromImage(sitk.ReadImage(segm_file))

            # Load distance map information 
            dist_file = os.path.join(dist_folder, f"{cid}.nii.gz") 
            dist_map = sitk.GetArrayFromImage(sitk.ReadImage(dist_file))
            
            # Iterate through boxes
            out_scores = []
            if boxes.shape[0] > 0:   
                # for i in range(boxes.shape[0]):
                #     out_score = box_acceptance(box = boxes[i], score=scores[i], 
                #                                dist = dist_map, time_map = time_map,
                #                                k=k)
                #     out_scores.append(out_score)
                _, collateral = extract_late_vessels(time_img= time_map)
                out_scores = Parallel(n_jobs=6)(delayed(box_acceptance)(boxes[i], scores[i], dist_map, time_map, collateral, k) for i in range(boxes.shape[0]))
                out_scores = np.array(out_scores)
                keep = np.ones(out_scores.shape[0], dtype=bool)
                ind_remove = np.where(keep == 0)
                keep[ind_remove] = False
            else:
                shutil.copyfile(pred_file, outfile)
                continue

            boxes_filtered, scores_filtered, labels_filtered = filter_out_fps(boxes = boxes, 
                                                                              scores=out_scores, 
                                                                              labels=labels, 
                                                                              keep=keep) 
            # boxes_filtered, scores_filtered, labels_filtered = boxes[keep] , scores[keep], labels[keep]

            out_dict["pred_boxes"] = boxes_filtered 
            out_dict["pred_labels"] = labels_filtered 
            out_dict["pred_scores"] = scores_filtered 

            write_data(data = out_dict, filename=outfile)


def get_args():
    # Remove predictions outside of lately enhanced regions 
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="Prediction folder", type=str)
    parser.add_argument("--dist", help="Distance map folder", type=str)
    parser.add_argument("--segm", help="Time map folder", type=str)
    parser.add_argument("--out", help="Output folder with FPR", type=str)
    parser.add_argument("--k", help="k time points to be sampled in boxes outside the vasculature", type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
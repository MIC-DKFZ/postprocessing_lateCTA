# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os, sys
import argparse
import SimpleITK as sitk
from nndet.core.boxes.ops_np import box_iou_np
from typing import Union
from scipy.spatial import cKDTree
from scipy.ndimage import label, binary_erosion, binary_dilation
from scipy.stats import rankdata
from joblib import Parallel, delayed
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import load_data, write_data
from utils.vessel_skeleton import (
    find_endpoints,
    compute_tip_image,
    skeletonization,
    dilate_skeleton,
    derive_patch_coords,
    smooth_time,
    filter_out_fps,
)


def get_closest_point(mask: np.ndarray, point: np.ndarray) -> Union[float, np.ndarray]:
    """
    Get closest point in mask to a query point


    Params
    ------
    mask : mask of interest
    point : query point




    Returns
    -------
    dist : distance of closest point
    coord : coordinate of closest point


    """
    # Coordinates of mask of interest
    coords = np.argwhere(mask > 0)
    tree = cKDTree(coords)

    # Query the k closest points
    dist, indices = tree.query(point, k=1)

    coord = coords[indices]

    return dist, coord


def label_endpoint(
    label_img: np.ndarray,
    endpoint: np.ndarray,
    time_img: np.ndarray,
    dist_others: int = 5,
) -> Union[int, float, bool]:
    """
    Provide connected component label to each endpoint found


    Params
    ------
    label_img : connected component labels
    endpoint : endpoint coordinate
    dist_others : minimum distance considered
        for the endpoint to be away from other components
    time_img : time vessel image
    time_mask : masked time vessel image


    Returns
    -------
    l : label from connected component
    t : closest time value
    close_to_others : endpoint close to other components


    """
    # Obtain label from connected component image
    l = label_img[endpoint[0], endpoint[1], endpoint[2]]
    closest = None
    if l == 0:
        # Endpoint not present in connected components
        _, closest = get_closest_point(mask=label_img, point=endpoint)

        l = label_img[closest[0], closest[1], closest[2]]

    # Obtain closest time value
    t = time_img[endpoint[0], endpoint[1], endpoint[2]]
    if t == 0:
        # Try with closest point from connected components
        if closest is not None:
            t = time_img[closest[0], closest[1], closest[2]]

        if t == 0:
            # Obtain closest point from time image
            _, closest_time = get_closest_point(mask=time_img, point=endpoint)
            t = time_img[closest_time[0], closest_time[1], closest_time[2]]

    # Obtain distance to other connected components
    label_others = label_img.copy()
    label_others[label_img == l] = 0
    d, _ = get_closest_point(mask=label_others, point=endpoint)
    close_to_others = True if d < dist_others else False

    return l, t, close_to_others


def exclude_small_ccs(img: np.ndarray, cc_img: np.ndarray, size_thr: int) -> np.ndarray:
    """
    Obtain size of connected components


    Params
    ------
    img : input image
    cc_img : input connected component image
    size_thr : size limit for very small connected components


    Returns
    -------
    out_img : image without small connected components


    """
    # Find sizes of connected components
    sizes = np.bincount(cc_img.ravel())

    # Create a mask for components larger than min_size
    mask = sizes > size_thr
    mask[0] = 0  # background stays 0

    # Use the mask to filter the labeled array
    out_img = mask[cc_img]

    return out_img


def distance2segm(dist_map: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Obtain segmentation from distance map,
    and preprocess it


    Params
    ------
    dist_map : distance map
    cfg : configuration


    Returns
    -------
    segm : segmentation


    """
    # Obtain segmentation
    segm = (dist_map <= cfg["distance_thr"]).astype(np.uint8)

    # Remove small connected components
    ccs, _ = label(segm)
    segm = exclude_small_ccs(img=segm, cc_img=ccs, size_thr=cfg["min_size_cc_segm"])

    # Remove noise
    if cfg["erosion_iters"] > 0:
        segm = binary_erosion(segm, iterations=cfg["erosion_iters"]).astype(np.uint8)
    segm = segm.astype(np.uint8)

    return segm


def locate_close_point_pairs(points: np.ndarray, thr: float) -> np.ndarray:
    """
    Locate close point pairs


    Params
    ------
    points : point set
    thr : threshold to consider points closeby




    Returns
    -------
    inds : close pair indexes


    """
    tree = cKDTree(points)
    close_pairs = list(tree.query_pairs(r=thr))
    inds = np.unique(np.array(close_pairs))
    return inds


def endpoint_analysis(
    patch: np.ndarray, endpoints: np.ndarray
) -> Union[np.ndarray, np.ndarray]:
    """
    Analyze endpoints in patch. If they belong to the same
    component and touch the borders of the patch, they
    form a continuous vessel, hence remove them


    Params
    ------
    patch : patch of interest
    endpoints : endpoints found in patch


    Return
    ------
    keep : binary array telling which endpoints to keep or discard
    cc : connected component of each endpoint


    """
    # Dilate patch before connected components to thicken skeleton
    struct = np.ones([3, 3, 3])
    patch = binary_dilation(patch, structure=struct, iterations=2).astype(np.uint8)
    cc_img, _ = label(patch, structure=struct)

    # Get cc for each endpoint
    cc, border = [], []
    for i in range(endpoints.shape[0]):
        # Obtain connected component
        cc.append(cc_img[endpoints[i, 0], endpoints[i, 1], endpoints[i, -1]])

        # Obtain border membership
        lower = (endpoints[i] == 0).any()  # Endpoint in lower border
        upper = np.array(patch.shape) - 1 - endpoints[i]
        upper = (upper == 0).any()  # Endpoint in upper border

        border.append(lower or upper)

    # Identify any border endpoints belonging to the same component
    border = np.array(border, dtype=bool)
    cc = np.array(cc)
    cc_remove = None
    keep = np.ones(endpoints.shape[0], dtype=bool)
    if border.any():
        ind = np.where(border)[0]
        cc_border = cc[ind]
        unique, count = np.unique(cc_border, return_counts=True)
        ind_count = np.where(count > 1)[0]
        if ind_count.shape[0] > 0:
            # there are connected components for continuous vessels in the patch
            # Discard all the endpoints in this connected component
            cc_remove = unique[ind_count]  # Connected components to be removed

    if cc_remove is not None:
        _, ind_remove, _ = np.intersect1d(cc, cc_remove, return_indices=True)
        keep[ind_remove] = False

    return keep, cc


def ccs_in_patch(patch: np.ndarray, min_size: int) -> bool:
    """
    Determine if there are any connected components inside the patch
    not touching borders


    Params
    ------
    patch : input patch
    min_size : minimum size of internal component


    Returns
    -------
    inside : whether there are full connected components
        inside patch (True) or not (False)


    """
    # Obtain connected component image
    cc_img, _ = label(input=patch)

    # Obtain border kernel
    kernel = np.ones(cc_img.shape)
    kernel[0, :, :] = 0
    kernel[-1, :, :] = 0
    kernel[:, 0, :] = 0
    kernel[:, -1, :] = 0
    kernel[:, :, 0] = 0
    kernel[:, :, -1] = 0

    # Iterate through components
    unique = np.unique(cc_img)[1:]

    inside = False
    i = 0
    while not (inside) and (i < unique.shape[0]):
        c = (cc_img == unique[i]).astype(int)
        if c.sum() > min_size:
            inside = c.sum() == (c * kernel).sum()
        i += 1

    return inside


def keep_patch(
    patch: np.ndarray, cfg: dict, time_patch: np.ndarray, time_thrs: list
) -> bool:
    """
    Based on local skeleton analysis, keep patch or not for FPR


    Params
    ------
    patch : patch to be analyzed
    cfg : configuration
    time_patch : patch from time image
    time_thrs : minimum and maximum times found in patch to consider it




    Returns
    -------
    keep : keep patch if True else False


    """

    keep = False  # Discard by default boxes outside the skeleton
    if patch.sum() > 0:
        # If there is some full connected component inside
        # the patch and does not touch the borders, keep the box
        # There may be some vessel inside
        inside = ccs_in_patch(patch=patch, min_size=cfg["min_size_cc_in_patch"])

        endpoints = find_endpoints(skeleton=patch)
        keep = True

        # Filter box if time is very early or very late
        time_vals_patch = time_patch[time_patch > 0].flatten()
        t = np.median(time_vals_patch)

        time_filter = True if (t < time_thrs[0]) or (t > time_thrs[-1]) else False

        if (endpoints.shape[0] > 0) and (not (inside)) and (not (time_filter)):
            # Discard any endpoints in the same connected component which are all in the border
            k, cc = endpoint_analysis(patch=patch, endpoints=endpoints)
            endpoints, cc = endpoints[k], cc[k]

            if endpoints.shape[0] > 0:
                # Discard any rows with zeros (endpoint in beginning of patch)
                ind_zeros = np.where(endpoints == 0)[0]

                if ind_zeros.shape[0] > 0:
                    non_zero_rows = np.setdiff1d(
                        np.arange(endpoints.shape[0]), ind_zeros
                    )
                    endpoints, cc = endpoints[non_zero_rows], cc[non_zero_rows]

                if endpoints.shape[0] > 0:
                    # Find if there are any endpoints touching the opposite box corners
                    ind_corners = []
                    for i in range(3):
                        ind_corner = np.where(endpoints[:, i] == (patch.shape[i] - 1))[
                            0
                        ]
                        ind_corners += ind_corner.tolist()
                    corner_rows = np.setdiff1d(
                        np.arange(endpoints.shape[0]), ind_corners
                    )
                    endpoints, cc = endpoints[corner_rows], cc[corner_rows]

                # Remove close point pairs: broken skeletonization
                if endpoints.shape[0] > 0:
                    close_inds = locate_close_point_pairs(
                        points=endpoints, thr=cfg["thr_close_points"]
                    )

                    if close_inds.shape[0] > 0:
                        # Obtain connected components of close endpoints
                        cc_close = cc[close_inds]
                        cc_unique, cc_count = np.unique(cc_close, return_counts=True)
                        # Preserve endpoints from the same connected component
                        keep_inds = np.setdiff1d(
                            np.arange(endpoints.shape[0]), close_inds
                        )
                        if (cc_count > 1).any():
                            k = np.where(cc_count > 1)[0]
                            keep_inds = np.concatenate([keep_inds, k])
                        endpoints, cc = endpoints[keep_inds], cc[keep_inds]
            else:
                keep = False

        if (endpoints.shape[0] == 0) or (time_filter):
            keep = False

        if inside:
            keep = True

    return keep


def derive_rank_image(img: np.ndarray) -> np.ndarray:
    """
    Derive rank image


    Params
    ------
    img : input image


    Returns
    -------
    rank : output rank image


    """
    mask = img > 0
    foreground_vals = img[mask].flatten()
    ranks = rankdata(foreground_vals, method="average")
    percentiles = (ranks - 1) * 100 / (len(foreground_vals) - 1)
    rank = np.zeros(img.shape, dtype=np.float32)
    rank[mask] = percentiles

    return rank


def process_case(
    file: os.PathLike,
    cfg: dict,
    pred_folder: os.PathLike,
    tta_folder: os.PathLike,
    brain_folder: os.PathLike,
    out_folder: os.PathLike,
    gt_folder: os.PathLike,
):
    """
    Process case


    Params
    ------
    file : prediction file
    cfg : configuration
    pred_folder : prediction folder
    segm_folder : time folder with segmentation information
    dist_folder : distance map folder
    out_folder : output folder
    gt_folder : ground-truth folder


    Returns
    -------
    Updated prediction file with preserved boxes


    """
    # Load prediction information
    cid = file.replace("_boxes.pkl", "")

    pred_file = os.path.join(pred_folder, file)
    pred = load_data(pred_file)
    out_dict = pred.copy()
    boxes, scores, labels = pred["pred_boxes"], pred["pred_scores"], pred["pred_labels"]

    outfile = os.path.join(out_folder, file)

    if not (os.path.exists(outfile)):
        print(cid)

        # Load time information
        tta_file = os.path.join(tta_folder, f"{cid}_0001.nii.gz")
        assert os.path.exists(
            tta_file
        ), f"Time-vessel map file '{tta_file}' does not exist"
        time_image = sitk.ReadImage(tta_file)
        time_map = sitk.GetArrayFromImage(time_image)

        # Smooth time information
        _, time_cc = smooth_time(
            time_map=time_map,
            size_thr=cfg["size_thr"],
            median_filter_size=cfg["median_filter_size"],
        )

        # Determine brain mask and centroid
        brain_file = os.path.join(brain_folder, f"{cid}.nii.gz")
        brain_image = sitk.ReadImage(brain_file)
        spacing = np.array(brain_image.GetSpacing())
        brain_segm = sitk.GetArrayFromImage(brain_image)
        brain_coords = np.argwhere(brain_segm > 0)
        brain_centroid = brain_coords.mean(axis=0).astype(int)
        brain_centroid_physical = brain_image.TransformContinuousIndexToPhysicalPoint(
            brain_centroid.tolist()
        )

        # Derive size of bounding box surrounding brain
        min_coords = np.min(brain_coords, 0)
        max_coords = np.max(brain_coords, 0)
        brain_size = (max_coords - min_coords) * spacing

        # Skeletonization
        skeleton, _ = skeletonization(segm=time_cc, image_ref=brain_image)

        # Dilate skeleton and expand it
        if cfg["expand_skeleton"] == 1:
            dilated_skeleton = dilate_skeleton(skeleton=skeleton)
            skeleton, _ = skeletonization(segm=dilated_skeleton, image_ref=brain_image)

        # Obtain mask with skeleton tips
        tips = find_endpoints(skeleton=skeleton)
        # Compute dynamic tip radius
        tip_mask = compute_tip_image(
            endpoints=tips, shape=skeleton.shape, radius=cfg["tip_radius"]
        )

        keep_box = []

        # Obtain TP mask for comparison with TP info
        tp_file = os.path.join(gt_folder, f"{cid}_boxes_gt.npz")
        tp_boxes = np.load(tp_file)["boxes"]
        tp_mask = np.zeros(time_map.shape, dtype=np.uint8)
        initial_tps = 0
        if tp_boxes.shape[0] > 0:

            for i in range(tp_boxes.shape[0]):
                tp_box = np.expand_dims(tp_boxes[i], 0)
                ious = box_iou_np(tp_box, boxes)
                ind_tps = np.where(ious >= 0.1)[0]
                if ind_tps.shape[0] > 0:
                    initial_tps += 1

                tp_box = tp_box.squeeze().astype(int)
                tp_mask[
                    tp_box[0] : tp_box[2],
                    tp_box[1] : tp_box[3],
                    tp_box[-2] : tp_box[-1],
                ] = 1

        # Iterate through each box
        tp_info, no_tip, no_lowdiam, no_rank = [], [], [], []
        scores_removed = []
        for box, score, l in zip(boxes, scores, labels):
            # Derive patch of interest
            coords = derive_patch_coords(box=box, shape=skeleton.shape)
            center = (
                np.array(
                    [
                        coords[0] + coords[2],
                        coords[1] + coords[3],
                        coords[-2] + coords[-1],
                    ]
                )
                // 2
            )
            center_physical = brain_image.TransformContinuousIndexToPhysicalPoint(
                center.tolist()
            )
            patch_tip = tip_mask[
                coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            ]

            patch_tp = tp_mask[
                coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            ]

            time_patch = time_map[
                coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            ]

            time_vals = time_patch[time_patch > 0]
            max_time = 0.0
            if time_vals.shape[0] > 0:
                # Get rank values
                # max_time = np.percentile(time_vals, 95)
                max_time = np.percentile(time_vals, 5)

            # time_condition = max_time > cfg["t_low_percentile"]
            time_condition = (max_time > cfg["t_low_percentile"]) & (
                max_time < cfg["t_high_percentile"]
            )

            keep = (patch_tip.sum() > 0).any() & time_condition

            # Determine causes for the removal of a positive
            empty_tip = (patch_tip.sum() == 0).all()
            empty_rank = not (time_condition)

            no_tip.append(empty_tip)
            no_rank.append(empty_rank)

            if patch_tp.sum() > 0:
                tp_info.append(True)
            else:
                tp_info.append(False)

            if (center.shape[0] > 0) and keep:
                # Compute relative location of prediction
                # Too high or too low positives tend to be FPs
                box_vector = (
                    np.array(center_physical) - np.array(brain_centroid_physical)
                ) / (brain_size + np.finfo(float).eps)

                if (
                    (box_vector[0] < cfg["relative_brain_pos"][0])
                    or (box_vector[0] > cfg["relative_brain_pos"][1])
                    or (box_vector[1] < cfg["relative_brain_pos"][2])
                    or (box_vector[1] > cfg["relative_brain_pos"][3])
                ):
                    keep = False

            if keep:
                vol = (box[2] - box[0]) * (box[3] - box[1]) * (box[-1] - box[-2]) / 1000

                if (vol < 1) or (vol > 200):
                    keep = False

            if not (keep):
                scores_removed.append(score)

            keep_box.append(keep)

        keep_box = np.array(keep_box, dtype=bool)

        # Forensics for TP removal
        final_tps = 0
        if initial_tps > 0:
            for i in range(tp_boxes.shape[0]):
                tp_box = np.expand_dims(tp_boxes[i], 0)
                ious = box_iou_np(tp_box, boxes[keep_box])
                ind_tps = np.where(ious >= 0.1)[0]
                if ind_tps.shape[0] > 0:
                    final_tps += 1

        if final_tps - initial_tps < 0:
            print(f"{cid} : WARNING: TP(s) removed!!")

        ious = box_iou_np(tp_boxes, boxes)
        tp_info = np.array(tp_info, dtype=bool)
        ind_tp = np.where(tp_info)[0]

        tp_removed = False
        message = ""
        if ind_tp.shape[0] > 0:
            # find if all TPs have been removed
            keep_tp = keep_box[ind_tp]
            if keep_tp.sum() == 0:
                tp_removed = True
                # Find the cause of the removal
                no_tip = np.array(no_tip, dtype=bool)
                no_lowdiam = np.array(no_lowdiam, dtype=bool)
                no_rank = np.array(no_rank, dtype=bool)
                no_tip, no_rank = (
                    no_tip[ind_tp],
                    no_rank[ind_tp],
                )

                if no_tip.sum() == no_tip.shape[0]:
                    message += "no tip found, "
                if no_rank.sum() == no_rank.shape[0]:
                    message += "no time information found"

        # Construct filtered results
        out_boxes, out_scores, out_labels, keep_box = filter_out_fps(
            boxes=boxes, scores=scores, labels=labels, keep=keep_box
        )

        out_dict["pred_boxes"] = out_boxes
        out_dict["pred_labels"] = out_labels
        out_dict["pred_scores"] = out_scores

        scores_removed = np.array(scores_removed)
        mean_scores_removed = 0
        if scores_removed.shape[0] > 0:
            mean_scores_removed = scores_removed.mean()

        if tp_removed:
            print(
                f"{cid} : Conserved fraction: {out_boxes.shape[0]*100/(boxes.shape[0] + np.finfo(float).eps)}%, TP removed: {tp_removed}, {message}, mean score removed: {mean_scores_removed}, score before: {scores.mean()},  score after: {out_scores.mean()}"
            )
        else:
            print(
                f"{cid} : Conserved fraction: {out_boxes.shape[0]*100/(boxes.shape[0] + np.finfo(float).eps)}%, mean score removed: {mean_scores_removed}, score before: {scores.mean()},  score after: {out_scores.mean()}"
            )

        write_data(data=out_dict, filename=outfile)


def main(args):
    pred_folder = args.pred
    tta_folder = args.segm
    out_folder = args.out
    brain_folder = args.brain
    gt_folder = args.ref

    assert os.path.exists(
        pred_folder
    ), f"Prediction folder '{pred_folder}' does not exist"
    assert os.path.exists(
        gt_folder
    ), f"Ground-truth folder '{gt_folder}' does not exist"
    assert os.path.exists(brain_folder), f"Brain folder '{brain_folder}' does not exist"
    assert os.path.exists(tta_folder), f"TTA folder '{tta_folder}' does not exist"
    assert os.path.exists(
        os.path.dirname(out_folder)
    ), f"Parent output folder '{out_folder}' does not exist"

    if not (os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Load configuration file
    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fpr_cfg.json")
    assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist"
    cfg = load_data(cfg_file)

    # Iterate through prediction files
    files = sorted(os.listdir(pred_folder))
    Parallel(n_jobs=cfg["workers"])(
        delayed(process_case)(
            file,
            cfg,
            pred_folder,
            tta_folder,
            brain_folder,
            out_folder,
            gt_folder,
        )
        for file in files
        if ".pkl" in file
    )


def get_args():
    # Remove predictions outside of lately enhanced regions
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="Prediction folder", type=str)
    parser.add_argument("--tta", help="Time map folder", type=str)
    parser.add_argument("--brain", help="Brain folder", type=str)
    parser.add_argument("--out", help="Output folder with FPR", type=str)
    parser.add_argument("--ref", help="Ground-truth folder", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

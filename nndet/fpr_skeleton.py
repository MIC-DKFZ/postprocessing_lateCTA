import numpy as np
import os, sys
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
from nndet.core.ops_np import box_iou_np
from typing import Union
from scipy.spatial import cKDTree, KDTree
from scipy.ndimage import (
    label,
    convolve,
    binary_erosion,
    binary_dilation,
    generate_binary_structure,
    generic_filter,
    median_filter,
    distance_transform_edt,
)
from scipy.stats import rankdata
from joblib import Parallel, delayed
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import load_data, write_data


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """
    Find skeleton endpoints

    Params
    ------
    skeleton : input skeleton where to find endpoints

    Returns
    -------
    endpoints : coordinates with endpoints

    """
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(skeleton.astype(int), kernel, mode="constant")
    endpoints = (skeleton == 1) & (neighbor_count == 1)
    return np.argwhere(endpoints)


def compute_tip_image(endpoints: np.ndarray, shape: tuple, radius: int = 15):
    """
    Create image out of all the endpoints found

    Params
    ------
    endpoints : tip points found from vessel skeletonization
    shape : image dimension
    radius : box radius to generate

    Returns
    -------
    tip : image with tip information

    """
    # Initialize image
    tip = np.zeros(shape)

    # Convert each point into a box centered around it
    start, end = endpoints - radius, endpoints + radius

    # Clip start and end coordinates to image dimensions
    start = np.clip(start, 0, None)
    for enum_s, s in enumerate(shape):
        end[:, enum_s] = np.clip(end[:, enum_s], 0, s - 1)

    # Iterate through endpoints
    for s, e in zip(start, end):
        tip[s[0] : e[0], s[1] : e[1], s[2] : e[2]] = 1

    return tip


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


def skeletonization(segm: np.ndarray, image_ref) -> np.ndarray:
    """
    Skeletonize from distance map

    Params
    ------
    segm : segmentation
    image_ref : reference SimpleITK image

    Returns
    -------
    skeleton : skeleton
    skeleton_image : SimpleITK skeleton image

    """
    segm_image = sitk.GetImageFromArray(segm)
    segm_image.CopyInformation(image_ref)
    segm_image = sitk.Cast(segm_image, sitk.sitkUInt8)
    skeleton_image = sitk.BinaryThinning(segm_image)
    skeleton = sitk.GetArrayFromImage(skeleton_image)

    return skeleton, skeleton_image


def dilate_skeleton(skeleton: np.ndarray):
    """
    Dilate skeleton to connected disconnected vessel fragments

    Params
    ------
    skeleton : skeleton
    cfg : configuration

    Returns
    -------
    dilated : dilated skeleton

    """
    # Connect the components: create a rather connected skeleton
    struct = generate_binary_structure(3, 2)

    # Dilate slightly
    dilated = binary_dilation(skeleton, structure=struct, iterations=1).astype(np.uint8)

    return dilated


def derive_patch_coords(box: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Derive coordinates of patch surrounding box of interest
    The patch consists of an area twice the size of the original box

    Output format:
    x0, y0, xf, yf, z0, zf

    Params
    ------
    box : box coordinates
    shape : image shape

    Returns
    -------
    coords : coordinates

    """
    # Derive height, width, and depth
    h, w, d = box[2] - box[0], box[3] - box[1], box[-1] - box[-2]
    corner = np.array([box[0] - 0.5 * h, box[1] - 0.5 * w, box[-2] - 0.5 * d])
    # Set coordinates to integer and ensure they fit the image size
    corner = np.ceil(np.clip(corner, 0, None)).astype(int)
    limits = corner + 2 * np.array([h, w, d])
    limits = np.array(
        [np.clip(l, 0, shape[i] - 1) for i, l in enumerate(limits)], dtype=int
    )

    coords = [corner[0], corner[1], limits[0], limits[1], corner[2], limits[2]]

    return coords


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


def smooth_time(
    time_map: np.ndarray, size_thr: int = 1000, median_filter_size: int = 5
) -> Union[np.ndarray, np.ndarray]:
    """
    Smooth time map and remove small connected components
    with a size below that 95 percentile of the sizes of
    the components found

    Params
    ------
    time_map : time map
    size_thr : size thresholding (default: 1000)
    median_filter_size : kernel size for median filter (default: 5)

    Returns
    -------
    smoothed : smoothed time information
    label_mask : mask with connected component labels

    """
    # Obtain connected components
    time_mask = time_map > np.finfo(float).eps
    label_time, _ = label(time_mask)

    # Count voxel occurrences for each label
    sizes = np.bincount(label_time.ravel())

    # Determine size threshold
    # size_thr = np.percentile(sizes, q)
    labels = np.arange(len(sizes))  # label numbers: 0, 1, 2, ..., num_features

    # Exclude background (label 0)
    sizes = sizes[1:]
    labels = labels[1:]
    labels = labels[sizes >= size_thr]

    if labels.shape[0] == 0:
        # Too harsh filtering applied for connected components
        # Apply directly map filtering
        smoothed = median_filter(input=time_map, size=median_filter_size)
        label_mask = (time_map > 0).astype(np.uint8)

        return smoothed, label_mask

    # Iterate through the greatest connected components
    smoothed = np.zeros(time_map.shape)
    label_mask = np.zeros(time_map.shape)
    for l in labels:
        if l != 0:
            smoothed[label_time == l] = time_map[label_time == l]
            label_mask[label_time == l] = l

    # smoothed_mask = (smoothed > 0).astype(int)

    # Apply median filter to time image
    # smoothed = apply_masked_median_filter(image=smoothed,
    #                                       mask=smoothed_mask,
    #                                       size=median_filter_size)
    smoothed = median_filter(input=smoothed, size=median_filter_size)

    return smoothed, label_mask


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
    segm_folder: os.PathLike,
    brain_folder: os.PathLike,
    dist_folder: os.PathLike,
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
        segm_file = os.path.join(segm_folder, f"{cid}.nii.gz")
        assert os.path.exists(
            segm_file
        ), f"Segmentation file '{segm_file}' does not exist"
        time_image = sitk.ReadImage(segm_file)
        time_map = sitk.GetArrayFromImage(time_image)

        # Smooth time information
        time_smooth, time_cc = smooth_time(
            time_map=time_map,
            size_thr=cfg["size_thr"],
            median_filter_size=cfg["median_filter_size"],
        )

        # Load distance map information
        dist_file = os.path.join(dist_folder, f"{cid}.nii.gz")
        assert os.path.exists(
            dist_file
        ), f"Distance map file '{dist_file}' does not exist"
        dist_image = sitk.ReadImage(dist_file)
        dist_map = sitk.GetArrayFromImage(dist_image)

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
        skeleton, _ = skeletonization(segm=time_cc, image_ref=dist_image)
        # Obtain connected components
        # time_mask = time_map > np.finfo(float).eps
        # time_cc, _ = label(time_mask)
        # skeleton, _ = skeletonization(segm=time_cc, image_ref=dist_image)

        # Dilate skeleton and expand it
        if cfg["expand_skeleton"] == 1:
            dilated_skeleton = dilate_skeleton(skeleton=skeleton)
            skeleton, _ = skeletonization(segm=dilated_skeleton, image_ref=dist_image)

        # Obtain diameter of skeleton
        diameter_map = skeleton * dist_map * 2

        # Rank time information
        # skeleton_time = skeleton * time_smooth  # Information on skeleton and time
        skeleton_time = skeleton * time_map
        # rank = derive_rank_image(img=skeleton_time)

        # Obtain mask with skeleton tips
        tips = find_endpoints(skeleton=skeleton)
        tip_mask = compute_tip_image(
            endpoints=tips, shape=skeleton.shape, radius=cfg["tip_radius"]
        )

        # Obtain mask with low diameter values
        # Minimum diameter found in foreground
        diameter_map = -diameter_map
        minimum_diam = np.percentile(
            diameter_map[diameter_map > 0], cfg["perc_min_diameter"]
        )
        low_diam_img = ((diameter_map > 0) & (diameter_map <= minimum_diam)).astype(
            np.uint8
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
            # coords = box.astype(int)
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
            patch_diam = low_diam_img[
                coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            ]
            patch_tp = tp_mask[
                coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            ]
            # time_patch = rank[
            #    coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            # ]
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

            keep = (
                (patch_tip.sum() > 0).any()
                & (patch_diam.sum() > 0).any()
                & time_condition
            )

            # Determine causes for the removal of a positive
            empty_tip = (patch_tip.sum() == 0).all()
            empty_low_diam = (patch_diam.sum() == 0).all()
            empty_rank = not (time_condition)

            no_tip.append(empty_tip)
            no_lowdiam.append(empty_low_diam)
            no_rank.append(empty_rank)

            if patch_tp.sum() > 0:
                tp_info.append(True)
            else:
                tp_info.append(False)

            # if (patch_tp.sum() > 0) and not (keep):
            #    print(
            #        cid,
            #        (patch_tip.sum() > 0).any(),
            #        (patch_diam.sum() > 0).any(),
            #        time_condition,
            #    )

            if (center.shape[0] > 0) and keep:
                # Compute relative location of prediction
                # Too high or too low positives tend to be FPs
                box_vector = (center_physical - brain_centroid_physical) / (
                    brain_size + np.finfo(float).eps
                )

                print(box_vector[0], box_vector[1], patch_tp.sum() > 0)

                if (
                    (box_vector[0] < -0.3)
                    or (box_vector[0] > 0.1)
                    or (box_vector[1] < -0.35)
                    or (box_vector[1] > 0.0)
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
                no_tip, no_lowdiam, no_rank = (
                    no_tip[ind_tp],
                    no_lowdiam[ind_tp],
                    no_rank[ind_tp],
                )

                if no_tip.sum() == no_tip.shape[0]:
                    message += "no tip found, "
                if no_lowdiam.sum() == no_lowdiam.shape[0]:
                    message += "no low diameter found, "
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


def filter_out_fps(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, keep: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
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

    if ratio < 0.05:
        # If less than 10% of boxes are remaining, we may be removing true positives
        # Keep the boxes with the top scores
        ind_remove = np.where(keep == False)[0]
        scores_remove = scores[ind_remove]
        scores_argsort = np.argsort(scores_remove)

        # Keep only 50% of the boxes
        n_remove = ind_remove.shape[0] // 2
        ind_scores_remove = scores_argsort[: (-1 - n_remove)]
        ind_remove_final = ind_remove[ind_scores_remove]
        keep = np.ones(keep.shape[0], dtype=bool)
        keep[ind_remove_final] = False

    out_boxes, out_scores, out_labels = boxes[keep], scores[keep], labels[keep]

    return out_boxes, out_scores, out_labels, keep


def main(args):
    pred_folder = args.pred
    segm_folder = args.segm
    dist_folder = args.dist
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
    assert os.path.exists(
        segm_folder
    ), f"Segmentation folder '{segm_folder}' does not exist"
    assert os.path.exists(
        dist_folder
    ), f"Distance map folder '{dist_folder}' does not exist"
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
            segm_folder,
            brain_folder,
            dist_folder,
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
    parser.add_argument("--dist", help="Distance map folder", type=str)
    parser.add_argument("--segm", help="Time map folder", type=str)
    parser.add_argument("--brain", help="Brain folder", type=str)
    parser.add_argument("--out", help="Output folder with FPR", type=str)
    parser.add_argument("--ref", help="Ground-truth folder", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

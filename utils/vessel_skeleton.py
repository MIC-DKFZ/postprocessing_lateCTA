# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

"""
Vessel skeleton and time-vessel map operations shared by the post-processing
scripts (fpr_skeleton.py and postprocess_end2end.py)
"""

from typing import Union

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    label,
    convolve,
    binary_dilation,
    generate_binary_structure,
    median_filter,
)


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
    # NOTE: dtypes matter a lot here. skeleton.astype(int) makes an int64 copy
    # (8 bytes/voxel) and a float64 kernel promotes the convolution output to float64
    # (another 8 bytes/voxel). scipy.ndimage.convolve returns the input dtype, and a
    # 3x3x3 neighbourhood can hold at most 26 neighbours, so uint8 is sufficient and
    # uses 1 byte/voxel instead of 16
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(
        skeleton.astype(np.uint8, copy=False), kernel, mode="constant"
    )
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
    # Initialize image (uint8: this is a binary mask, np.zeros would default to
    # float64 and use 8x the memory of a full-resolution volume for no benefit)
    tip = np.zeros(shape, dtype=np.uint8)

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
        # Apply directly map filtering
        smoothed = median_filter(
            input=time_map.astype(np.float32, copy=False), size=median_filter_size
        )
        label_mask = (time_map > 0).astype(np.uint8)

        return smoothed, label_mask

    # Keep only the connected components that survive the size threshold.
    # NOTE: this replaces a per-label Python loop that recomputed 'label_time == l'
    # twice per component (two full-volume boolean arrays per iteration). np.isin
    # builds a single mask instead. The output dtypes are also pinned: np.zeros
    # defaults to float64, which doubled the memory of both volumes for no benefit
    keep_mask = np.isin(label_time, labels)
    smoothed = np.where(keep_mask, time_map, 0).astype(np.float32, copy=False)
    label_mask = np.where(keep_mask, label_time, 0).astype(np.int32, copy=False)
    del keep_mask

    # Release the label volume before median_filter allocates its own output
    del label_time, time_mask
    smoothed = median_filter(input=smoothed, size=median_filter_size)

    return smoothed, label_mask


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
        # If less than 5% of boxes are remaining, we may be removing true positives
        # Keep the boxes with the top scores. Avoid complete removal of boxes
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

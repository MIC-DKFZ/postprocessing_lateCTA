import os, sys
import numpy as np
import time
import argparse
from joblib import Parallel, delayed
from batchgenerators.utilities.file_and_folder_operations import load_json
from nndet.io.load import load_pickle
import SimpleITK as sitk
from scipy.ndimage import (
    convolve,
    binary_dilation,
    generate_binary_structure,
    median_filter,
    label,
)
from scipy.spatial import cKDTree
import h5py
from typing import Union


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


def process_box(
    box: np.ndarray,
    tta: np.ndarray,
    centroid: list,
    tips: np.ndarray,
    brain_image,
    brain_size: tuple,
):
    """
    Process box information

    Params
    ------
    box : box coordinates (x0, y0, xf, yf, z0, zf)
    tta : time-vessel map
    centroid : physical brain centroid
    tips : coordinates where the vessels in the time vessel map get interrupted
    brain_image : brain image
    brain_size : brain size

    Returns
    -------
    d : distance to nearest tip
    t : time value for box
    p : whether box lies in posterior brain or not

    """
    # Extract patch around box
    center = 0.5 * np.array([box[0] + box[2], box[1] + box[3], box[-2] + box[-1]])
    patch = derive_patch_coords(box=box, shape=tta.shape)

    # Obtain distance to tips
    tree = cKDTree(tips)
    d, _ = tree.query(center)

    # Obtain time values
    time_patch = tta[patch[0] : patch[2], patch[1] : patch[3], patch[-2] : patch[-1]]
    t = 1.0
    p = 0
    if time_patch.sum() > 0:
        time_vals = time_patch[time_patch > 0]
        t = np.percentile(time_vals, 5)
        center_physical = brain_image.TransformContinuousIndexToPhysicalPoint(
            center.tolist()
        )

        box_vector = (np.array(center_physical) - np.array(centroid)) / (
            brain_size + np.finfo(float).eps
        )

        if (
            (box_vector[0] < -0.3)
            or (box_vector[0] > 0.1)
            or (box_vector[1] < -0.35)
            or (box_vector[1] > 0.1)
        ):
            p = 1

    return d, t, p


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


def process_case(
    predfolder: os.PathLike,
    timefolder: os.PathLike,
    brainfolder: os.PathLike,
    thr: float,
    file: os.PathLike,
    workers: int = 4,
):
    """
    Process case

    Params
    ------
    predfolder : prediction folder
    timefolder : time vessel map folder
    brainfolder : brain segmentation folder
    thr : confidence threshold
    file : prediction file
    workers : number of parallel workers

    """

    full_file = os.path.join(predfolder, file)
    cid = file.replace("_boxes.pkl", "")

    # Load prediction information
    pred = load_pickle(full_file)
    boxes, scores, labels = (
        pred["pred_boxes"],
        pred["pred_scores"],
        pred["pred_labels"],
    )
    pred["dists"], pred["times"], pred["posterior"] = (
        np.array([]),
        np.array([]),
        np.array([]),
    )

    # Discard boxes based on scores
    if boxes.shape[0] > 0:
        ind = np.where(scores >= thr)[0]
        if ind.shape[0]:
            boxes, scores, labels = boxes[ind], scores[ind], labels[ind]

        if boxes.shape[0] > 0:
            # Load brain segmentation
            brainfile = os.path.join(brainfolder, f"{cid}.nii.gz")
            brain_image = sitk.ReadImage(brainfile)
            spacing = np.array(brain_image.GetSpacing())
            brain_segm = sitk.GetArrayFromImage(brain_image)
            brain_coords = np.argwhere(brain_segm > 0)
            brain_centroid = brain_coords.mean(axis=0).astype(int)
            brain_centroid_physical = (
                brain_image.TransformContinuousIndexToPhysicalPoint(
                    brain_centroid.tolist()
                )
            )
            # Derive size of bounding box surrounding brain
            min_coords = np.min(brain_coords, 0)
            max_coords = np.max(brain_coords, 0)
            brain_size = (max_coords - min_coords) * spacing

            # Load time-vessel map
            time_file = os.path.join(timefolder, f"{cid}.nii.gz")

            tta_image = sitk.ReadImage(time_file)
            tta = sitk.GetArrayFromImage(tta_image)
            time_smooth, time_cc = smooth_time(
                time_map=tta,
                size_thr=1000,
                median_filter_size=5,
            )

            # Convert time-vessel map into skeleton and obtain endpoints
            skeleton, _ = skeletonization(segm=time_cc, image_ref=tta_image)
            dilated_skeleton = dilate_skeleton(skeleton=skeleton)
            # Overdilate skeleton
            skeleton, _ = skeletonization(segm=dilated_skeleton, image_ref=tta_image)
            # Extract tips
            tips = find_endpoints(skeleton=skeleton)
            # iterate through boxes
            data = Parallel(n_jobs=workers)(
                delayed(process_box)(
                    box,
                    time_smooth,
                    brain_centroid_physical,
                    tips,
                    brain_image,
                    brain_size,
                )
                for box in boxes
            )
            data = np.array(data)
            pred["dists"], pred["times"], pred["posterior"] = (
                np.array(data[:, 0]),
                np.array(data[:, 1]),
                np.array(data[:, 2]),
            )

    return {cid: pred}


def main(args):
    predfolder = args.p
    timefolder = args.t
    thr_file = args.thr
    workers = args.np
    outfile = args.o
    gtfolder = args.g
    brainfolder = args.b

    assert os.path.exists(
        predfolder
    ), f"Prediction folder '{predfolder}' does not exist"
    assert os.path.exists(
        timefolder
    ), f"Time vessel map folder '{timefolder}' does not exist"
    assert os.path.exists(thr_file), f"Threshold file '{thr_file}' does not exist"
    assert os.path.exists(brainfolder), f"Brain folder '{brainfolder}' does not exist"
    assert os.path.exists(gtfolder), f"Ground truth folder '{gtfolder}' does not exist"
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(
        ".h5"
    ), f"Parent output directory '{os.path.dirname(outfile)}' does not exist or is not .h5"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Threshold file loading
    thr = load_json(thr_file)["thr_box"]

    # Set up files
    files = sorted(os.listdir(predfolder))
    with h5py.File(outfile, "w") as f:
        for file in files:
            if file.endswith(".pkl"):
                cid = file.replace("_boxes.pkl", "")
                # Load ground truth file
                gt_file = os.path.join(gtfolder, f"{cid}_boxes_gt.npz")
                gt = np.load(gt_file, allow_pickle=True)["boxes"]
                group = f.create_group(cid)

                # Store values of distances, times and posterior locations for all predicted boxes
                # with a score over the optimal one
                d = process_case(
                    predfolder=predfolder,
                    timefolder=timefolder,
                    brainfolder=brainfolder,
                    thr=thr,
                    file=file,
                    workers=workers,
                )
                preds = list(d.values())[0]
                print(cid, d)
                group.create_dataset(
                    "pred_boxes",
                    data=preds["pred_boxes"],
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )
                group.create_dataset(
                    "pred_scores", data=preds["pred_scores"], compression="gzip"
                )
                group.create_dataset(
                    "pred_labels", data=preds["pred_labels"], compression="gzip"
                )
                group.create_dataset(
                    "pred_dists", data=preds["dists"], compression="gzip"
                )
                group.create_dataset(
                    "pred_times", data=preds["times"], compression="gzip"
                )
                group.create_dataset(
                    "pred_posterior", data=preds["posterior"], compression="gzip"
                )
                group.create_dataset("gt_boxes", data=gt, compression="gzip")


def get_args():
    # Obtain prediction fingerprint based on false occlusion removal rules
    parser = argparse.ArgumentParser(description="Obtain prediction fingerprint")
    parser.add_argument("--p", help="Prediction folder", required=True, type=str)
    parser.add_argument("--t", help="Time vessel map folder", required=True, type=str)
    parser.add_argument("--thr", help="Threshold file", required=True, type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    parser.add_argument("--b", help="Brain folder", required=True, type=str)
    parser.add_argument("--g", help="Ground truth folder", required=True, type=str)
    parser.add_argument("--np", help="Number of parallel workers", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

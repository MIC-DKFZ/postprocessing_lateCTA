import os, sys
import numpy as np
import time
import argparse
from joblib import Parallel, delayed
from skimage.morphology import skeletonize
from batchgenerators.utilities.file_and_folder_operations import load_json
import SimpleITK as sitk
from scipy.ndimage import (
    convolve,
    binary_dilation,
    generate_binary_structure,
    median_filter,
    label,
    distance_transform_edt,
    maximum_filter,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial import cKDTree
import h5py
from typing import Union
from nndet.io.load import load_pickle
from nndet.core.ops_np import box_iou_np
import skfmm
from scipy.spatial.distance import cdist


def extract_skeleton_break_candidates(
    segm: np.ndarray,
) -> np.ndarray:
    """
    Extract vessel skeleton break candidates in voxel (image) space.

    Parameters
    ----------
    segm : np.ndarray
        Binary vessel segmentation (ZYX).
    distance_map : np.ndarray
        Distance transform of segmentation (same shape as segm, in voxels).
    radius_thr_vox : float
        Minimum vessel radius (in voxels) to consider endpoint valid.
    max_pair_dist_vox : float
        Maximum distance (in voxels) between two endpoints to form a break.

    Returns
    -------
    break_points_vox : np.ndarray (N, 3)
        Break candidate coordinates in voxel space (ZYX).
    """

    # 1️⃣ Skeletonize in 3D
    skeleton = skeletonize(segm.astype(bool))
    # dilated_skeleton = dilate_skeleton(skeleton)
    # skeleton = skeletonize(dilated_skeleton.astype(bool))

    # Detect endpoints using 26-neighborhood
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    endpoints = (skeleton == 1) & (neighbor_count == 1)
    # branches = (skeleton == 1) & (neighbor_count >= 3)

    # endpoints = np.logical_or(endpoints, branches)

    endpoint_vox = np.argwhere(endpoints)
    if len(endpoint_vox) < 2:
        return np.empty((0, 3), dtype=float)

    return np.vstack(endpoint_vox)


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
    tree: np.ndarray,
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
            (box_vector[0] < -0.4)
            or (box_vector[0] > 0.2)
            or (box_vector[1] < -0.4)
            or (box_vector[1] > 0.2)
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

    smoothed = median_filter(input=smoothed, size=median_filter_size)

    return smoothed, label_mask


def process_case(
    predfolder: os.PathLike,
    timefolder: os.PathLike,
    brainfolder: os.PathLike,
    thr: float,
    file: os.PathLike,
    gt: np.ndarray,
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
    gt : ground-truth box coordinates
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
    pred["tp"] = np.array([])

    # Discard boxes based on scores
    if boxes.shape[0] > 0:
        ind = np.where(scores >= thr)[0]
        if ind.shape[0] > 0:
            boxes, scores, labels = boxes[ind], scores[ind], labels[ind]
            pred["pred_boxes"], pred["pred_scores"], pred["pred_labels"] = (
                boxes,
                scores,
                labels,
            )

            # Load brain segmentation
            brainfile = os.path.join(brainfolder, f"{cid}.nii.gz")
            brain_image = sitk.ReadImage(brainfile)
            spacing = np.flip(np.array(brain_image.GetSpacing()))
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

            # Convert time-vessel map into skeleton and obtain endpoints
            tips = extract_skeleton_break_candidates(segm=tta)
            tree = cKDTree(tips)  # Extract tree for distance computations

            # iterate through boxes
            data = [
                process_box(
                    box, tta, brain_centroid_physical, tree, brain_image, brain_size
                )
                for box in boxes
            ]

            data = np.array(data)
            pred["dists"], pred["times"], pred["posterior"] = (
                np.array(data[:, 0]),
                np.array(data[:, 1]),
                np.array(data[:, 2]),
            )

            # Gather true positive labels
            iou = box_iou_np(boxes, gt)
            tp_inds = np.where(iou >= 0.10)[0]
            tps = np.zeros(boxes.shape[0])
            tps[tp_inds] = 1
            pred["tp"] = tps

    return {cid: pred}


def process_single_case(args_tuple):
    """
    Worker function for multiprocessing.
    Must be top-level (pickle requirement).
    """
    (
        file,
        gtfolder,
        predfolder,
        timefolder,
        brainfolder,
        thr,
    ) = args_tuple

    try:
        if not file.endswith(".pkl"):
            return None

        cid = file.replace("_boxes.pkl", "")

        # Load GT
        gt_file = os.path.join(gtfolder, f"{cid}_boxes_gt.npz")
        gt = np.load(gt_file, allow_pickle=True)["boxes"]

        # Run heavy processing
        d = process_case(
            predfolder=predfolder,
            timefolder=timefolder,
            brainfolder=brainfolder,
            thr=thr,
            file=file,
            gt=gt,
        )
        preds = list(d.values())[0]

        return cid, preds, gt

    except Exception as e:
        print(f"[ERROR] Failed case {file}: {e}")
        return None


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", required=True)
    parser.add_argument("--t", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--g", required=True)
    parser.add_argument("--o", required=True)
    parser.add_argument("--thr", required=True)
    parser.add_argument("--np", type=int, default=os.cpu_count())

    args = parser.parse_args()

    files = sorted(os.listdir(args.p))

    # load threshold file
    thr = load_json(args.thr)["thr_box"]

    # Prepare arguments for workers
    worker_args = [
        (
            file,
            args.g,
            args.p,
            args.t,
            args.b,
            thr,
        )
        for file in files
        if file.endswith(".pkl")
    ]

    results = []

    print(f"Processing {len(worker_args)} cases with {args.np} workers...\n")

    # -----------------------------
    # Parallel Processing
    # -----------------------------
    with ProcessPoolExecutor(max_workers=args.np) as executor:
        futures = [executor.submit(process_single_case, wa) for wa in worker_args]

        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res is not None:
                results.append(res)

    print(f"\nFinished processing. Writing HDF5 file...\n")

    # -----------------------------
    # Sequential HDF5 Writing
    # -----------------------------
    with h5py.File(args.o, "w") as f:
        for cid, preds, gt in results:

            group = f.create_group(cid)

            group.create_dataset(
                "pred_boxes",
                data=preds["pred_boxes"],
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            group.create_dataset(
                "pred_scores",
                data=preds["pred_scores"],
                compression="gzip",
            )
            group.create_dataset(
                "pred_labels",
                data=preds["pred_labels"],
                compression="gzip",
            )
            group.create_dataset(
                "pred_dists",
                data=preds["dists"],
                compression="gzip",
            )
            group.create_dataset(
                "pred_times",
                data=preds["times"],
                compression="gzip",
            )
            group.create_dataset(
                "pred_posterior",
                data=preds["posterior"],
                compression="gzip",
            )
            group.create_dataset(
                "tp",
                data=preds["tp"],
                compression="gzip",
            )
            group.create_dataset(
                "gt_boxes",
                data=gt,
                compression="gzip",
            )

    print("\nDone.")


# --------------------------------------------------
if __name__ == "__main__":
    main()

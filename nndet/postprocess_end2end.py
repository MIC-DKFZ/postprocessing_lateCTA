import os, sys
import SimpleITK as sitk
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor_regression
import argparse
from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
)
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import pandas as pd
from typing import Union
import time
from scipy.ndimage import (
    label,
    convolve,
    binary_dilation,
    generate_binary_structure,
    median_filter,
)
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.load_save import load_data, write_data

# TotalSegmentator lives in its OWN environment (it pins nnunetv2 from PyPI, which
# would collide with the custom nnunetv2 fork used by this script). It is therefore
# invoked as an external process. Point TOTALSEG_BIN at the executable, e.g.
#   export TOTALSEG_BIN=/path/to/envs/totalseg/bin/TotalSegmentator
# or pass --totalseg_bin on the command line.
DEFAULT_TOTALSEG_BIN = os.environ.get("TOTALSEG_BIN", "TotalSegmentator")

# numpy index arrays are in (z, y, x) order, while SimpleITK's
# TransformContinuousIndexToPhysicalPoint and GetSpacing use (x, y, z). The original
# R3 rule mixed the two. Setting this to True corrects the ordering, but it changes
# the meaning of the R3 cutoffs (box_vector thresholds) in the configuration file,
# so those must be retuned. Leave it False to reproduce previously tuned thresholds.
FIX_AXIS_ORDER = False

# Print the raw R3 quantities for every kept box (debugging aid)
VERBOSE_R3 = False

# Suffix identifying the first (and only) input channel of a case
CHANNEL_SUFFIX = "_0000.nii.gz"


def cid_from_file(file: os.PathLike) -> str:
    """Derive the case identifier from an image filename"""
    return os.path.basename(str(file)).replace(CHANNEL_SUFFIX, "")


def load_case_ids(id_file: os.PathLike = "") -> set:
    """
    Read the set of case identifiers to process from a TXT file


    One identifier per line. Blank lines and lines starting with '#' are ignored.
    Identifiers may be written bare ("case001") or with the image suffix
    ("case001_0000.nii.gz", "case001.nii.gz"); both are normalised to the bare id.


    Params
    ------
    id_file : path to the TXT file, or an empty string to disable filtering


    Returns
    -------
    ids : set of case identifiers, or None if no filtering is requested

    """
    if id_file is None or len(str(id_file).strip()) == 0:
        return None

    assert os.path.exists(id_file), f"ID file '{id_file}' does not exist"
    assert str(id_file).endswith(".txt"), f"ID file '{id_file}' is not a .txt file"

    # NOTE: read line by line rather than with np.loadtxt. A one-line file makes
    # np.loadtxt return a 0-d array whose .tolist() is a bare string, and 'cid in ids'
    # would then silently become a substring test instead of a membership test
    with open(id_file, "r") as f:
        raw = [line.strip() for line in f]

    ids = set()
    for entry in raw:
        if (len(entry) == 0) or entry.startswith("#"):
            continue
        ids.add(cid_from_file(entry).replace(".nii.gz", ""))

    assert len(ids) > 0, f"ID file '{id_file}' contains no usable identifiers"
    return ids


def select_files(data_folder: os.PathLike, ids: set = None) -> list:
    """
    List the case files to process, optionally restricted to a set of identifiers


    Params
    ------
    data_folder : folder with CTA images
    ids : set of case identifiers to keep, or None to keep everything


    Returns
    -------
    files : sorted list of filenames to process
    missing : sorted list of requested identifiers with no matching file

    """
    files = sorted(f for f in os.listdir(data_folder) if f.endswith(CHANNEL_SUFFIX))

    if ids is None:
        return files, []

    keep = [f for f in files if cid_from_file(f) in ids]
    missing = sorted(ids - {cid_from_file(f) for f in keep})
    return keep, missing


class StageTimer:
    """
    Collect wall-clock timings per processing stage, per case


    Usage
    -----
    timer = StageTimer()
    with timer("inference", cid):
        ...
    timer.summary()

    """

    def __init__(self, sync_cuda: bool = True):
        self.records = []  # list of {"cid", "stage", "seconds"}
        self.sync_cuda = sync_cuda

    def _sync(self):
        # CUDA kernels are launched asynchronously: without a sync, a GPU stage
        # would report only the time spent queueing the work
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def __call__(self, stage: str, cid: str = "-"):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.record(stage=stage, cid=cid, seconds=time.perf_counter() - t0)

    def record(self, stage: str, cid: str, seconds: float) -> None:
        """Add a timing measured outside of the context manager"""
        self.records.append({"cid": cid, "stage": stage, "seconds": seconds})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["cid", "stage", "seconds"])

    def summary(self, total_seconds: float = None, n_cases: int = None) -> pd.DataFrame:
        """Aggregate per stage and print a table. Returns the aggregated frame."""
        df = self.to_frame()
        if df.empty:
            print("No timings recorded")
            return df

        agg = (
            df.groupby("stage")["seconds"]
            .agg(calls="count", total="sum", mean="mean", median="median", max="max")
            .sort_values("total", ascending=False)
        )
        agg["percent"] = 100.0 * agg["total"] / agg["total"].sum()

        print("\n" + "=" * 72)
        print("EXECUTION TIME BREAKDOWN")
        print("=" * 72)
        with pd.option_context("display.float_format", lambda v: f"{v:10.2f}"):
            print(agg.to_string())
        print("-" * 72)
        if total_seconds is not None:
            print(f"{'Total wall clock':<28}{fmt_hms(total_seconds)}")
            if n_cases:
                print(
                    f"{'Per case':<28}{fmt_hms(total_seconds / n_cases)}"
                    f"  ({n_cases} cases)"
                )
        print("=" * 72 + "\n", flush=True)
        return agg


def fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.mmm"""
    h, rem = divmod(float(seconds), 3600.0)
    m, s = divmod(rem, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def run_totalsegmentator(
    in_file: os.PathLike,
    out_folder: os.PathLike,
    totalseg_bin: str = DEFAULT_TOTALSEG_BIN,
    fast: bool = False,
    device: str = "gpu",
) -> None:
    """
    Run TotalSegmentator (task "total") as an external process and write the
    brain mask to <out_folder>/brain.nii.gz


    Params
    ------
    in_file : input CTA (NIfTI)
    out_folder : folder where TotalSegmentator writes its output
    totalseg_bin : path to the TotalSegmentator executable (separate environment)
    fast : use the 3mm model instead of the 1.5mm one
    device : "gpu", "cpu" or "gpu:X"


    Returns
    -------
    None (writes <out_folder>/brain.nii.gz)

    """
    cmd = [
        totalseg_bin,
        "-i",
        str(in_file),
        "-o",
        str(out_folder),
        "--task",
        "total",
        "--roi_subset",
        "brain",
        # roi_subset crops with a 6mm model that is unreliable on head-only FOVs;
        # robust_crop uses the slower but safer 3mm model instead
        "--robust_crop",
        "--device",
        device,
        "--quiet",
    ]
    if fast:
        cmd.append("--fast")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator failed on '{in_file}' (exit {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )


def segment_brain(
    in_file: os.PathLike,
    cache_folder: os.PathLike,
    cid: str,
    totalseg_bin: str = DEFAULT_TOTALSEG_BIN,
    fast: bool = False,
    device: str = "gpu",
) -> sitk.Image:
    """
    Return a binary brain mask (sitk.Image) in the same space as in_file,
    computed with TotalSegmentator and cached on disk


    Params
    ------
    in_file : input CTA (NIfTI)
    cache_folder : folder where brain masks are cached as <cid>.nii.gz
    cid : case identifier
    totalseg_bin : path to the TotalSegmentator executable (separate environment)
    fast : use the 3mm model instead of the 1.5mm one
    device : "gpu", "cpu" or "gpu:X"


    Returns
    -------
    brain : binary brain mask in the geometry of in_file

    """
    os.makedirs(cache_folder, exist_ok=True)
    mask_file = os.path.join(cache_folder, f"{cid}.nii.gz")

    # Reuse the cached mask if it already exists
    if os.path.exists(mask_file):
        return sitk.ReadImage(mask_file)

    # Per-case temporary folder: TotalSegmentator always names its output
    # "brain.nii.gz", so a shared folder would let cases overwrite each other
    tmp_out = tempfile.mkdtemp(prefix=f"totalseg_{cid}_")
    try:
        run_totalsegmentator(
            in_file=in_file,
            out_folder=tmp_out,
            totalseg_bin=totalseg_bin,
            fast=fast,
            device=device,
        )
        brain_file = os.path.join(tmp_out, "brain.nii.gz")
        if not os.path.exists(brain_file):
            raise RuntimeError(
                f"TotalSegmentator did not produce a brain mask for '{cid}'"
            )
        brain = sitk.ReadImage(brain_file)
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)

    brain = sitk.Cast(brain > 0, sitk.sitkUInt8)
    if sitk.GetArrayViewFromImage(brain).sum() == 0:
        raise RuntimeError(f"Empty brain mask for '{cid}'")

    sitk.WriteImage(brain, mask_file, True)
    return brain


def precompute_brain_masks(
    data_folder: os.PathLike,
    cache_folder: os.PathLike,
    files: list,
    totalseg_bin: str = DEFAULT_TOTALSEG_BIN,
    fast: bool = False,
    device: str = "gpu",
    timer: "StageTimer" = None,
) -> None:
    """
    Segment the brain for every selected case before the main loop starts.

    Running this as a separate pass keeps TotalSegmentator and the CTA-to-time-vessel
    map predictor from holding GPU memory at the same time, and pays the model
    loading cost once per case instead of interleaving it with inference.


    Params
    ------
    data_folder : folder with CTA images
    cache_folder : folder where brain masks are cached as <cid>.nii.gz
    files : filenames to process, already filtered by select_files
    totalseg_bin : path to the TotalSegmentator executable (separate environment)
    fast : use the 3mm model instead of the 1.5mm one
    device : "gpu", "cpu" or "gpu:X"
    timer : optional StageTimer collecting per-case timings


    Returns
    -------
    None (populates cache_folder)

    """
    for i, file in enumerate(files):
        cid = cid_from_file(file)

        # Distinguish a real segmentation from a cache hit, otherwise the timing
        # summary is meaningless on re-runs
        cached = os.path.exists(os.path.join(cache_folder, f"{cid}.nii.gz"))
        stage = "totalsegmentator_cached" if cached else "totalsegmentator"

        t0 = time.perf_counter()
        segment_brain(
            in_file=os.path.join(data_folder, file),
            cache_folder=cache_folder,
            cid=cid,
            totalseg_bin=totalseg_bin,
            fast=fast,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        if timer is not None:
            timer.record(stage=stage, cid=cid, seconds=elapsed)

        print(
            f"[TotalSegmentator {i + 1}/{len(files)}] {cid} "
            f"({'cached' if cached else 'segmented'}, {elapsed:.2f}s)",
            flush=True,
        )


def postprocess_preds(
    pred: np.ndarray, params: dict, brain: np.ndarray = None
) -> np.ndarray:
    """
    Postprocess predictions with postprocessing
    parameters

    Params
    ------
    pred : prediction from original translation model
    params : postprocessing parameters (cutoff for distance, time values)
    brain : brain segmentation for case of interest

    Returns
    -------
    out : postprocessed output prediction

    """
    # Threshold distance map
    dist_thr = (pred[0] <= float(params["dist"])).astype(np.uint8)

    # Threshold segmentation map
    segm_thr = (pred[-1] >= float(params["logit"])).astype(np.uint8)

    # Union map
    out = ((dist_thr + segm_thr) > 0).astype(np.uint8)

    out = (out * pred[1]).astype(
        np.float32
    )  # Map with only time information in vessels

    # Clip map to avoid having times > 1 and negative times
    out = np.clip(out, a_min=0.0, a_max=1.0)

    # Restrict map to brain area, if brain segmentation is available
    if brain is not None:
        out *= brain

    return out.astype(np.float32)


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

    # smoothed_mask = (smoothed > 0).astype(int)

    # Apply median filter to time image
    # smoothed = apply_masked_median_filter(image=smoothed,
    #                                       mask=smoothed_mask,
    #                                       size=median_filter_size)
    # Release the label volume before median_filter allocates its own output
    del label_time, time_mask
    smoothed = median_filter(input=smoothed, size=median_filter_size)

    return smoothed, label_mask


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


def process_case(
    file: os.PathLike,
    cfg: dict,
    image,
    time_map: np.ndarray,
    brain_segm: np.ndarray,
    out_folder: os.PathLike,
):
    """
    Process case


    Params
    ------
    file : prediction file
    cfg : loaded configuration dictionary (not the path to the .json)
    image: SimpleITK CTA image being processed
    time_map: time-vessel map (from previous step)
    brain_segm: brain mask from TotalSegmentator as a numpy array, (z, y, x)
    out_folder : output folder


    Returns
    -------
    Updated prediction file with preserved boxes


    """
    # Load prediction information
    pred = load_data(file)
    out_dict = pred.copy()
    boxes, scores, labels = pred["pred_boxes"], pred["pred_scores"], pred["pred_labels"]

    outfile = os.path.join(out_folder, os.path.basename(file))

    if not (os.path.exists(outfile)):  # Skip if file has already been postprocessed

        # Smooth time information
        _, time_cc = smooth_time(
            time_map=time_map,
            size_thr=cfg["size_thr"],
            median_filter_size=cfg["median_filter_size"],
        )

        # Set brain mask and centroid for R3
        spacing = np.array(image.GetSpacing())  # SimpleITK order: (x, y, z)
        brain_coords = np.argwhere(brain_segm > 0)  # numpy order: (z, y, x)
        brain_centroid = brain_coords.mean(axis=0)

        # Derive size of bounding box surrounding brain
        min_coords = np.min(brain_coords, 0)
        max_coords = np.max(brain_coords, 0)
        brain_extent = max_coords - min_coords  # numpy order: (z, y, x)

        if FIX_AXIS_ORDER:
            brain_centroid_index = brain_centroid[::-1]
            brain_size = brain_extent[::-1] * spacing
        else:
            brain_centroid_index = brain_centroid
            brain_size = brain_extent * spacing

        brain_centroid_physical = image.TransformContinuousIndexToPhysicalPoint(
            [float(c) for c in brain_centroid_index]
        )

        # Skeletonization of vessels in smoothed time-vessel map
        skeleton, _ = skeletonization(segm=time_cc, image_ref=image)

        # Dilate skeleton and expand it,
        # if specified in configuration, to avoid small vessel segmentation breaks
        if cfg["expand_skeleton"] == 1:
            dilated_skeleton = dilate_skeleton(skeleton=skeleton)
            skeleton, _ = skeletonization(segm=dilated_skeleton, image_ref=image)

        # Obtain mask with skeleton tips for R1
        tips = find_endpoints(skeleton=skeleton)
        # Compute dynamic tip radius
        tip_mask = compute_tip_image(
            endpoints=tips, shape=skeleton.shape, radius=cfg["tip_radius"]
        )

        keep_box = []

        # Iterate through each box
        for box, score, l in zip(boxes, scores, labels):
            # Derive patch of interest around every box, enlarge twice box size
            coords = derive_patch_coords(box=box, shape=skeleton.shape)
            # Derive center of patch
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
            # Derive centroid of box for R3 in physical coordinates
            center_index = center[::-1] if FIX_AXIS_ORDER else center
            center_physical = image.TransformContinuousIndexToPhysicalPoint(
                [float(c) for c in center_index]
            )
            # Set patches in tip mask and in time-vessel maps
            patch_tip = tip_mask[
                coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            ]
            time_patch = time_map[
                coords[0] : coords[2], coords[1] : coords[3], coords[-2] : coords[-1]
            ]

            # Set up time value threshold for patch of interest
            time_vals = time_patch[time_patch > 0]
            max_time = 0.0
            if time_vals.shape[0] > 0:
                max_time = np.percentile(time_vals, 5)

            time_condition = max_time < cfg["t_high_percentile"]

            # Combine R1 and R2 constraints
            keep = (patch_tip.sum() > 0).any() & time_condition

            if (center.shape[0] > 0) and keep:
                # Compute relative location of prediction
                # Too high or too low positives in superior and posterior brain tend to be FPs
                box_vector = (
                    np.array(center_physical) - np.array(brain_centroid_physical)
                ) / (
                    brain_size + np.finfo(float).eps
                )  # normalized box coordinate to brain centroid difference

                if (
                    (box_vector[0] < -0.3)
                    or (box_vector[0] > 0.1)
                    or (box_vector[1] < -0.35)
                    or (box_vector[1] > 0.1)
                ):
                    keep = False

                if keep:  # Remove boxes with very small or very large volumes
                    vol = (
                        (box[2] - box[0])
                        * (box[3] - box[1])
                        * (box[-1] - box[-2])
                        / 1000
                    )

                    if (vol < 1) or (vol > 200):
                        keep = False

            keep_box.append(keep)

        keep_box = np.array(keep_box, dtype=bool)

        # Construct filtered results
        out_boxes, out_scores, out_labels, keep_box = filter_out_fps(
            boxes=boxes, scores=scores, labels=labels, keep=keep_box
        )

        out_dict["pred_boxes"] = out_boxes
        out_dict["pred_labels"] = out_labels
        out_dict["pred_scores"] = out_scores

        write_data(data=out_dict, filename=outfile)


def main(args):
    run_t0 = time.perf_counter()  # end-to-end wall clock starts here
    timer = StageTimer()

    data_folder = args.d  # Folder with test images
    model_folder = args.m  # Folder with CTA-to-time-vessel map model information
    pred_folder = args.p  # Folder with original predictions to post-process
    out_folder = args.o  # Folder with post-processed estimations
    mode = args.mode  # CTA-to-time-vessel map model preference "final" or "best"
    cfg_file = args.cfg  # Configuration file with post-processing hyperparameters
    workers = args.np  # Parallel workers
    brain_cache = args.brain_cache  # Folder with cached TotalSegmentator brain masks
    totalseg_bin = args.totalseg_bin  # TotalSegmentator executable (separate env)
    totalseg_fast = args.totalseg_fast  # Use the 3mm TotalSegmentator model
    totalseg_device = args.totalseg_device  # Device for TotalSegmentator
    id_file = args.i  # file with specific IDs to postprocess

    assert os.path.exists(data_folder), f"Data folder '{data_folder}' does not exist"
    assert os.path.exists(model_folder), f"Model folder '{model_folder}' does not exist"
    assert os.path.exists(
        pred_folder
    ), f"Prediction folder '{pred_folder}' does not exist"
    assert os.path.exists(
        os.path.dirname(out_folder)
    ), f"Parent output folder '{os.path.dirname(out_folder)}' does not exist"
    assert mode.lower().strip() in ["final", "best"]
    assert (os.path.exists(cfg_file)) and (
        cfg_file.endswith(".json")
    ), f"Configuration file '{cfg_file}' does not exist or is not .json"
    assert workers > 0, "Zero or negative number of parallel workers"
    assert shutil.which(totalseg_bin) is not None, (
        f"TotalSegmentator executable '{totalseg_bin}' not found. Install it in its own "
        "environment and pass --totalseg_bin /path/to/env/bin/TotalSegmentator "
        "(or set the TOTALSEG_BIN environment variable)"
    )

    # Create output folder if it does not exist
    if not (os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Load postprocessing configuration (process_case expects a dict, not a path)
    cfg = load_json(cfg_file)

    # Resolve which cases to process ONCE, before any work is done, so that the brain
    # mask pre-pass and the main loop can never disagree about the selection
    case_ids = load_case_ids(id_file)
    files, missing = select_files(data_folder=data_folder, ids=case_ids)

    assert len(files) > 0, f"No cases to process in '{data_folder}'" + (
        f" matching the IDs in '{id_file}'" if case_ids is not None else ""
    )
    if case_ids is None:
        print(f"Processing all {len(files)} cases in '{data_folder}'", flush=True)
    else:
        print(
            f"Processing {len(files)}/{len(case_ids)} requested cases "
            f"from '{id_file}'",
            flush=True,
        )
    if len(missing) > 0:
        print(
            f"WARNING: {len(missing)} requested ID(s) have no matching "
            f"'*{CHANNEL_SUFFIX}' file and will be skipped: "
            f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}",
            flush=True,
        )

    # Segment the brain of every selected case up front, in a separate process, so
    # that TotalSegmentator and the time-vessel map predictor never share GPU memory
    precompute_brain_masks(
        data_folder=data_folder,
        cache_folder=brain_cache,
        files=files,
        totalseg_bin=totalseg_bin,
        fast=totalseg_fast,
        device=totalseg_device,
        timer=timer,
    )

    # Load postprocessing parameters for time-vessel map estimation
    params_file = os.path.join(model_folder, "best_dist_logit_cutoff.json")
    assert os.path.exists(
        params_file
    ), f"Post processing parameter file '{params_file}' does not exist"
    params = load_json(params_file)

    # Initialize predictor for CTA-to-time-vessel map generator
    with timer("predictor_init"):
        predictor = nnUNetPredictor_regression(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda", 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=("all",),
            checkpoint_name=f"checkpoint_{mode.lower().strip()}.pth",
        )

    # iterate through the selected files: time-vessel map estimation + postprocessing
    for i, file in enumerate(files):
        cid = cid_from_file(file)
        full_file = os.path.join(data_folder, file)

        print(f"cid : {cid} ({i + 1}/{len(files)})", flush=True)
        case_t0 = time.perf_counter()

        with timer("read_image", cid):
            image = sitk.ReadImage(full_file)

        # Load brain information (already computed by precompute_brain_masks; this
        # only reads the cached mask, and recomputes it if the cache was cleared)
        with timer("read_brain_mask", cid):
            brain_img = segment_brain(
                in_file=full_file,
                cache_folder=brain_cache,
                cid=cid,
                totalseg_bin=totalseg_bin,
                fast=totalseg_fast,
                device=totalseg_device,
            )
            brain = sitk.GetArrayFromImage(brain_img).astype(np.uint8)

        with timer("prepare_input", cid):
            img, props = SimpleITKIO().read_images([full_file])
            assert brain.shape == img.shape[1:], (
                f"Brain mask shape {brain.shape} does not match image shape "
                f"{img.shape[1:]} for case '{cid}'"
            )
            ind = np.where(brain == 0)
            img[0, ind[0], ind[1], ind[2]] = (
                -1024.0
            )  # Set background outside brain vessels to -1024

        with timer("inference", cid):
            iterator = predictor.get_data_iterator_from_raw_npy_data(
                [img], None, [props], None, workers
            )
            r = predictor.predict_from_data_iterator(iterator, False, 1)

        # Postprocess time-vessel map prediction with distance, time, and segmentation heads
        with timer("postprocess_preds", cid):
            time_vessel_map = postprocess_preds(pred=r[0], params=params, brain=brain)

        # Set up prediction information
        pred_file = os.path.join(pred_folder, f"{cid}_boxes.pkl")
        if os.path.exists(pred_file):
            with timer("process_case", cid):
                process_case(
                    file=pred_file,
                    cfg=cfg,
                    image=image,
                    time_map=time_vessel_map,
                    brain_segm=brain,
                    out_folder=out_folder,
                )
        else:
            print(f"    no prediction file '{pred_file}', skipping", flush=True)

        print(f"    case time: {time.perf_counter() - case_t0:.2f}s", flush=True)

    # Report end-to-end execution time
    total_seconds = time.perf_counter() - run_t0
    timer.summary(total_seconds=total_seconds, n_cases=len(files))

    timings_file = os.path.join(out_folder, "timings.csv")
    timer.to_frame().to_csv(timings_file, index=False)
    print(f"Per-case timings written to '{timings_file}'", flush=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="Data folder (CTA)", required=True, type=str)
    parser.add_argument(
        "--m",
        help="Model folder (CTA-to-time-vessel map generator)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--p",
        help="Prediction folder (original nnDetection predictions)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--o", help="Output folder (postprocessed predictions)", required=True, type=str
    )
    parser.add_argument(
        "--mode",
        help="Checkpoint to evaluate (CTA-to-time-vessel map generator)",
        default="final",
        type=str,
    )
    parser.add_argument(
        "--cfg", help="Postprocessing configuration file", required=True, type=str
    )
    parser.add_argument("--np", help="Number of parallel workers", default=1, type=int)
    parser.add_argument(
        "--brain_cache",
        help="Folder where TotalSegmentator brain masks are cached (<cid>.nii.gz)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--totalseg_bin",
        help=(
            "Path to the TotalSegmentator executable, which must live in its own "
            "environment (defaults to $TOTALSEG_BIN, then to 'TotalSegmentator')"
        ),
        default=DEFAULT_TOTALSEG_BIN,
        type=str,
    )
    parser.add_argument(
        "--totalseg_fast",
        help="Use the 3mm TotalSegmentator model instead of the 1.5mm one",
        action="store_true",
    )
    parser.add_argument(
        "--totalseg_device",
        help="Device for TotalSegmentator ('gpu', 'cpu' or 'gpu:X')",
        default="gpu",
        type=str,
    )
    parser.add_argument(
        "--i",
        help=(
            "TXT file with the case IDs to process, one per line (blank lines and "
            "lines starting with '#' are ignored). IDs may be bare ('case001') or "
            "include the image suffix. Omit to process every case in the data folder"
        ),
        required=False,
        default="",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())

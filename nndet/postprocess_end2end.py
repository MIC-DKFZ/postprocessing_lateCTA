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
import atexit
import csv
import gc
import resource
import shutil
import signal
import subprocess
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime

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


# Label of the brain class in the TotalSegmentator "total" task class map
BRAIN_LABEL = 90

# Filename used for the multilabel segmentation of a full (non-roi_subset) run
ML_SEG_NAME = "segmentation.nii.gz"


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


FIELDS = [
    "run_id",
    "ts",
    "cid",
    "stage",
    "seconds",
    "rss_gb",
    "peak_rss_gb",
    "status",
    "error",
]


class StageTimer:
    """
    Collect wall-clock timings and resident memory per processing stage, per case

    Rows are appended to the CSV and flushed to disk as soon as each stage ends, so
    the record survives a hard kill (OOM killer, SIGKILL, node failure) where no
    traceback or atexit handler would ever run.


    Usage
    -----
    timer = StageTimer()
    timer.attach_csv("out/timings.csv")
    with timer("inference", cid):
        ...
    timer.summary()

    """

    def __init__(self, sync_cuda: bool = True):
        self.records = []  # list of dicts keyed by FIELDS
        self.sync_cuda = sync_cuda
        self._fh = None
        self._writer = None
        # Identifies this invocation. When several short-lived runs append to the
        # same CSV (e.g. with --max_cases), this is what separates them
        self.run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"

    def attach_csv(self, path: os.PathLike, resume: bool = True) -> None:
        """
        Stream every recorded row to 'path' as it happens

        With resume=True an existing file is appended to rather than truncated, so a
        series of short-lived runs (--max_cases) accumulates into a single timeline
        and the evidence from a run that crashed is preserved.
        """
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        mode = "a" if (exists and resume) else "w"

        if mode == "a":
            # Appending to a file written by an older version would silently misalign
            # every column, so verify the header matches before trusting it
            try:
                with open(path, "r", newline="") as f:
                    header = next(csv.reader(f), [])
            except Exception:
                header = []

            if header != FIELDS:
                backup = f"{path}.{datetime.now().strftime('%Y%m%dT%H%M%S')}.bak"
                shutil.move(path, backup)
                print(
                    f"NOTE: existing timings have a different column layout and were "
                    f"moved to '{os.path.basename(backup)}'; starting a new file",
                    flush=True,
                )
                exists, mode = False, "w"
            else:
                # A run killed mid-write can leave a partial final line. Append a
                # newline if needed so the next row does not fuse onto it
                try:
                    with open(path, "rb") as f:
                        f.seek(-1, os.SEEK_END)
                        needs_nl = f.read(1) not in (b"\n", b"\r")
                    if needs_nl:
                        with open(path, "a", newline="") as f:
                            f.write("\n")
                        print(
                            "NOTE: repaired a truncated final line in the timings file",
                            flush=True,
                        )
                except Exception:
                    pass

        # line buffered; every write is followed by an explicit flush + fsync
        self._fh = open(path, mode, newline="", buffering=1)
        self._writer = csv.DictWriter(
            self._fh, fieldnames=FIELDS, extrasaction="ignore"
        )
        if mode == "w" or not exists:
            self._writer.writeheader()
            self._fh.flush()

        # Emit anything recorded before the CSV was attached
        for rec in self.records:
            self._write_row(rec)

    def _write_row(self, rec: dict) -> None:
        if self._writer is None:
            return
        try:
            self._writer.writerow(rec)
            self._fh.flush()
            os.fsync(self._fh.fileno())  # push past the OS page cache
        except Exception as exc:  # never let logging kill the run
            print(f"WARNING: could not write timing row: {exc}", flush=True)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
            except Exception:
                pass
            self._fh, self._writer = None, None

    @staticmethod
    def rss_gb() -> float:
        """Current resident set size of this process, in GiB"""
        try:
            with open("/proc/self/statm", "r") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / 2**30
        except Exception:
            return float("nan")

    @staticmethod
    def peak_rss_gb() -> float:
        """Peak RSS of this process since it started, in GiB (monotonic)"""
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        except Exception:
            return float("nan")

    def _sync(self):
        # CUDA kernels are launched asynchronously: without a sync, a GPU stage
        # would report only the time spent queueing the work
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def __call__(self, stage: str, cid: str = "-"):
        self._sync()
        t0 = time.perf_counter()
        status, err = "ok", ""
        try:
            yield
        except BaseException as exc:
            # Record the partial timing before the exception propagates, so a stage
            # that dies still leaves a row behind
            status, err = "error", f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            try:
                self._sync()
            except Exception:
                pass
            self.record(
                stage=stage,
                cid=cid,
                seconds=time.perf_counter() - t0,
                status=status,
                error=err,
            )

    def record(
        self,
        stage: str,
        cid: str,
        seconds: float,
        status: str = "ok",
        error: str = "",
    ) -> None:
        """Add a timing and immediately persist it"""
        rec = {
            "run_id": self.run_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cid": cid,
            "stage": stage,
            "seconds": seconds,
            "rss_gb": self.rss_gb(),
            "peak_rss_gb": self.peak_rss_gb(),
            "status": status,
            "error": error,
        }
        self.records.append(rec)
        self._write_row(rec)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=FIELDS)

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


def count_children() -> int:
    """
    Number of live child processes of this process

    nnU-Net spawns a preprocessing worker pool per data iterator. If those workers
    are not reaped between cases they accumulate, and each one holds its own copy of
    the volume it was handed, so host memory grows case by case.
    """
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/task")) and len(
            [
                p
                for p in os.listdir("/proc")
                if p.isdigit() and _ppid_of(int(p)) == os.getpid()
            ]
        )
    except Exception:
        return -1


def _ppid_of(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            # field 4 is ppid; the comm field may contain spaces so split after ')'
            return int(f.read().rpartition(")")[2].split()[1])
    except Exception:
        return -1


def count_open_fds() -> int:
    """Number of open file descriptors, another thing that leaks across iterations"""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        return -1


def system_memory_gb() -> tuple:
    """Return (total, available) system memory in GiB, or (nan, nan)"""
    try:
        info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = float(rest.strip().split()[0]) / 2**20  # kB -> GiB
        return info.get("MemTotal", float("nan")), info.get(
            "MemAvailable", float("nan")
        )
    except Exception:
        return float("nan"), float("nan")


def check_memory_headroom(limit_gb: float, cid: str = "-") -> None:
    """
    Abort with a clear error if this process is close to a memory ceiling

    An OOM kill arrives as SIGKILL: no traceback, no cleanup, nothing written. This
    raises a normal MemoryError slightly before that point, so the failure is
    recorded and (with --keep_going) the run continues with the next case.
    """
    if limit_gb is None or limit_gb <= 0:
        return
    rss = StageTimer.rss_gb()
    if np.isfinite(rss) and rss > limit_gb:
        raise MemoryError(
            f"RSS {rss:.2f} GB exceeded the --mem_limit_gb ceiling of {limit_gb:.2f} GB "
            f"while processing '{cid}'. Lower --np, or process this case separately"
        )


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
    roi_subset: bool = False,
) -> None:
    """
    Run TotalSegmentator (task "total") as an external process


    With roi_subset=False (default) the full multilabel segmentation is written to
    <out_folder>/<ML_SEG_NAME> and the caller extracts label BRAIN_LABEL. With
    roi_subset=True only the brain class is predicted, into
    <out_folder>/brain.nii.gz, which is much faster but crops first


    Params
    ------
    in_file : input CTA (NIfTI)
    out_folder : folder where TotalSegmentator writes its output
    totalseg_bin : path to the TotalSegmentator executable (separate environment)
    fast : use the 3mm model instead of the 1.5mm one
    device : "gpu", "cpu" or "gpu:X"
    roi_subset : predict only the brain class (fast, but crops to an ROI first)


    Returns
    -------
    None (writes into out_folder)

    """
    if roi_subset:
        # Fast path: predict the brain class only. NOTE this crops to a region of
        # interest first, and that crop can cut the segmentation off at the edge.
        # robust_crop uses the slower 3mm crop model instead of the 6mm one, but the
        # crop still happens, so this path does NOT reproduce a full run exactly
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
            "--robust_crop",
            "--device",
            device,
            "--quiet",
        ]
    else:
        # Full run over every class, then extract the brain label. This is what
        # totalsegmentator(input_img) with no arguments does, so it reproduces masks
        # generated by the standalone segmentation script. Much slower
        cmd = [
            totalseg_bin,
            "-i",
            str(in_file),
            "-o",
            os.path.join(str(out_folder), ML_SEG_NAME),
            "--task",
            "total",
            "--ml",
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
    brain_folder: os.PathLike = "",
    roi_subset: bool = False,
) -> sitk.Image:
    """
    Return a binary brain mask (sitk.Image) in the same space as in_file


    If brain_folder is given and contains <cid>.nii.gz, that mask is used directly and
    TotalSegmentator is not run. This is what reproduces earlier experiments exactly:
    the R3 rule normalises by the bounding box of the mask, so even a handful of stray
    voxels changes brain_size by double-digit percentages and rescales box_vector for
    every box in the case.


    Params
    ------
    in_file : input CTA (NIfTI)
    cache_folder : folder where brain masks are cached as <cid>.nii.gz
    cid : case identifier
    totalseg_bin : path to the TotalSegmentator executable (separate environment)
    fast : use the 3mm model instead of the 1.5mm one
    device : "gpu", "cpu" or "gpu:X"
    brain_folder : folder with precomputed masks, taking priority over segmentation
    roi_subset : predict only the brain class (fast, but crops to an ROI first)


    Returns
    -------
    brain : binary brain mask in the geometry of in_file

    """
    # Precomputed masks win: no segmentation, no caching, no modification
    if len(str(brain_folder)) > 0:
        external = os.path.join(brain_folder, f"{cid}.nii.gz")
        if os.path.exists(external):
            brain = sitk.ReadImage(external)
            return sitk.Cast(brain > 0, sitk.sitkUInt8)

    os.makedirs(cache_folder, exist_ok=True)
    mask_file = os.path.join(cache_folder, f"{cid}.nii.gz")

    # Reuse the cached mask if it already exists
    if os.path.exists(mask_file):
        return sitk.ReadImage(mask_file)

    # Per-case temporary folder: TotalSegmentator uses fixed output names, so a
    # shared folder would let cases overwrite each other
    tmp_out = tempfile.mkdtemp(prefix=f"totalseg_{cid}_")
    try:
        run_totalsegmentator(
            in_file=in_file,
            out_folder=tmp_out,
            totalseg_bin=totalseg_bin,
            fast=fast,
            device=device,
            roi_subset=roi_subset,
        )

        if roi_subset:
            seg_file = os.path.join(tmp_out, "brain.nii.gz")
        else:
            seg_file = os.path.join(tmp_out, ML_SEG_NAME)
        if not os.path.exists(seg_file):
            raise RuntimeError(
                f"TotalSegmentator did not produce '{os.path.basename(seg_file)}' "
                f"for '{cid}'"
            )

        seg = sitk.ReadImage(seg_file)
        if roi_subset:
            # Single-class output: any positive voxel is brain
            brain = sitk.Cast(seg > 0, sitk.sitkUInt8)
        else:
            # Multilabel output: keep only the brain class, exactly as
            # (segm == 90) does in the standalone segmentation script
            brain = sitk.Cast(
                sitk.Equal(sitk.Cast(seg, sitk.sitkInt32), BRAIN_LABEL),
                sitk.sitkUInt8,
            )
        brain.CopyInformation(seg)
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)

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


def save_time_vessel_map(
    time_map: np.ndarray,
    image_ref,
    out_folder: os.PathLike,
    cid: str,
    suffix: str = "_0001.nii.gz",
) -> str:
    """
    Write the predicted time-vessel map to disk for comparison against the
    precomputed maps used to tune the post-processing thresholds


    The output is written in the geometry of image_ref (the CTA) and with the same
    naming convention as the precomputed maps (<cid>_0001.nii.gz), so both can be
    read back and compared voxel for voxel without any resampling.


    Params
    ------
    time_map : predicted time-vessel map, numpy order (z, y, x)
    image_ref : SimpleITK image defining spacing, origin and direction
    out_folder : folder to write into
    cid : case identifier
    suffix : filename suffix, matching the precomputed maps by default


    Returns
    -------
    outfile : path of the written map

    """
    os.makedirs(out_folder, exist_ok=True)
    outfile = os.path.join(out_folder, f"{cid}{suffix}")

    img = sitk.GetImageFromArray(time_map.astype(np.float32, copy=False))
    img.CopyInformation(image_ref)  # spacing, origin and direction of the CTA
    sitk.WriteImage(img, outfile, True)

    return outfile


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

            # R2: the time value must fall inside the enhancement window. Both bounds
            # are required to reproduce fpr_skeleton.py; dropping the lower one keeps
            # every early-enhancing box that the original rejected
            time_condition = (max_time > cfg["t_low_percentile"]) & (
                max_time < cfg["t_high_percentile"]
            )

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
                if VERBOSE_R3:
                    print(center_physical, brain_centroid_physical, brain_size)

                if (
                    (box_vector[0] < -0.3)
                    or (box_vector[0] > 0.1)
                    or (box_vector[1] < -0.35)
                    or (box_vector[1] > 0.1)
                ):
                    keep = False

            # R4: reject implausible box volumes (in mL, hence the /1000). Applied on
            # the raw box rather than the enlarged patch, matching fpr_skeleton.py.
            # The bounds were hardcoded as 1 and 200 in the original; they are read
            # from the config here if present, so the defaults reproduce it exactly
            if keep:
                vol = (box[2] - box[0]) * (box[3] - box[1]) * (box[-1] - box[-2]) / 1000

                if (vol < cfg.get("min_volume", 1)) or (
                    vol > cfg.get("max_volume", 200)
                ):
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
    keep_going = args.keep_going  # Continue after a failing case
    mem_limit_gb = args.mem_limit_gb  # RSS ceiling per case (0 disables)
    save_maps = args.save_maps  # Folder for predicted time-vessel maps ('' disables)
    max_cases = args.max_cases  # Process at most this many new cases (0 = all)
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

    # Start streaming timings to disk NOW. Every row is flushed and fsynced as soon
    # as its stage ends, so the file stays valid even if the process is killed
    # outright (OOM killer, scheduler, node failure) with no chance to clean up
    timings_file = os.path.join(out_folder, "timings.csv")
    timer.attach_csv(timings_file, resume=True)
    print(f"Streaming timings to '{timings_file}' (run_id {timer.run_id})", flush=True)

    if len(save_maps) > 0:
        os.makedirs(save_maps, exist_ok=True)
        print(
            f"Saving predicted time-vessel maps to '{save_maps}'. NOTE: cases whose "
            "output already exists are skipped before inference, so clear the output "
            "folder (or use a fresh one) to get a map for every case",
            flush=True,
        )

    mem_total, mem_avail = system_memory_gb()
    print(
        f"System memory: {mem_total:.1f} GB total, {mem_avail:.1f} GB available | "
        f"workers (--np) = {workers}"
        + (f" | ceiling {mem_limit_gb:.1f} GB" if mem_limit_gb > 0 else ""),
        flush=True,
    )

    # Print whatever was collected on the way out, for any exit path that still runs
    # Python: normal end, unhandled exception, sys.exit, SIGTERM or SIGINT
    def _final_report(reason: str = "exit"):
        if getattr(_final_report, "done", False):
            return
        _final_report.done = True
        print(f"\n--- run ended ({reason}) ---", flush=True)
        try:
            timer.summary(total_seconds=time.perf_counter() - run_t0, n_cases=None)
        except Exception as exc:
            print(f"Could not print summary: {exc}", flush=True)
        timer.close()

    atexit.register(_final_report, "atexit")

    def _on_signal(signum, frame):
        # SIGTERM is what a scheduler or 'kill' sends; SIGKILL cannot be caught,
        # which is exactly why the CSV is written incrementally rather than here
        _final_report(f"signal {signal.Signals(signum).name}")
        raise SystemExit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, _on_signal)
        except (ValueError, OSError, AttributeError):
            pass  # not available on this platform or not the main thread

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
    n_ok, n_failed, n_done, failed_cids = 0, 0, 0, []
    for i, file in enumerate(files):
        cid = cid_from_file(file)
        full_file = os.path.join(data_folder, file)

        # Skip completed cases BEFORE any work. process_case has its own check, but it
        # only fires after inference has already run, so a resumed run would repeat
        # ~60s of GPU work per finished case just to throw the result away
        expected_out = os.path.join(out_folder, f"{cid}_boxes.pkl")
        if os.path.exists(expected_out):
            print(
                f"cid : {cid} ({i + 1}/{len(files)}) already done, skipping", flush=True
            )
            n_done += 1
            continue

        # Stop after a fixed number of cases so a long run can be split across several
        # short-lived processes. Anything the process leaks is reclaimed by the OS on
        # exit, and the next invocation resumes from the skip check above
        if (max_cases > 0) and (n_ok + n_failed >= max_cases):
            print(
                f"\nReached --max_cases {max_cases}; exiting cleanly. "
                f"{len(files) - n_done - n_ok - n_failed} case(s) still to do",
                flush=True,
            )
            break

        print(f"cid : {cid} ({i + 1}/{len(files)})", flush=True)
        case_t0 = time.perf_counter()

        try:
            with timer("read_image", cid):
                image = sitk.ReadImage(full_file)

            # Load brain information (already computed by precompute_brain_masks;
            # this only reads the cached mask, recomputing it if the cache was cleared)
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

            check_memory_headroom(mem_limit_gb, cid)
            with timer("inference", cid):
                iterator = predictor.get_data_iterator_from_raw_npy_data(
                    [img], None, [props], None, workers
                )
                r = predictor.predict_from_data_iterator(iterator, False, 1)

            # Postprocess time-vessel map with distance, time and segmentation heads
            with timer("postprocess_preds", cid):
                time_vessel_map = postprocess_preds(
                    pred=r[0], params=params, brain=brain
                )

            # Optionally persist the predicted map for threshold recalibration
            if len(save_maps) > 0:
                with timer("save_time_vessel_map", cid):
                    save_time_vessel_map(
                        time_map=time_vessel_map,
                        image_ref=image,
                        out_folder=save_maps,
                        cid=cid,
                    )

            check_memory_headroom(mem_limit_gb, cid)

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

            n_ok += 1

        except Exception as exc:
            # One bad case should not throw away the whole run. The failure is
            # recorded in the CSV and printed with a full traceback, then the loop
            # moves on. MemoryError is included: it is a normal Python exception,
            # unlike an OOM kill, which never reaches this handler
            n_failed += 1
            failed_cids.append(cid)
            timer.record(
                stage="case_failed",
                cid=cid,
                seconds=time.perf_counter() - case_t0,
                status="error",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            if not keep_going:
                raise

        finally:
            # Drop references to the large per-case volumes before the next
            # iteration allocates its own, whether the case succeeded or not.
            # Rebinding to None is what actually releases them: del locals()[...]
            # would operate on a snapshot dict and free nothing
            img = r = time_vessel_map = brain = brain_img = image = None
            iterator = props = None
            gc.collect()

            print(
                f"    case time: {time.perf_counter() - case_t0:.2f}s"
                f" | RSS {timer.rss_gb():.2f} GB (peak {timer.peak_rss_gb():.2f} GB)"
                f" | children {count_children()} | fds {count_open_fds()}",
                flush=True,
            )

    # Report end-to-end execution time
    print(
        f"\nCompleted {n_ok} case(s) this run, {n_failed} failed, "
        f"{n_done} already done, {len(files)} selected",
        flush=True,
    )
    if failed_cids:
        print(f"Failed cases: {', '.join(failed_cids)}", flush=True)
        failed_file = os.path.join(out_folder, "failed_cids.txt")
        with open(failed_file, "w") as f:
            f.write("\n".join(failed_cids) + "\n")
        print(f"Retry them with --i '{failed_file}'", flush=True)

    _final_report("completed")
    print(f"Per-case timings written to '{timings_file}'", flush=True)

    # Machine-readable trailer for wrapper scripts: how many selected cases still
    # have no output. A chunked driver loop uses this to stop as soon as the work is
    # finished, instead of paying predictor_init for every remaining iteration
    remaining = sum(
        1
        for f in files
        if not os.path.exists(os.path.join(out_folder, f"{cid_from_file(f)}_boxes.pkl"))
    )
    print(f"REMAINING={remaining}", flush=True)


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
    parser.add_argument("--np", help="Number of parallel workers", default=4, type=int)
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
        "--save_maps",
        help=(
            "Folder to write the predicted time-vessel map of every case as "
            "<cid>_0001.nii.gz, in the CTA geometry. Use it to compare against the "
            "precomputed maps the post-processing thresholds were tuned on"
        ),
        default="",
        type=str,
    )
    parser.add_argument(
        "--max_cases",
        help=(
            "Process at most this many new cases, then exit cleanly (0 = all). Use it "
            "to split a run across several short-lived processes when memory grows "
            "case by case; finished cases are skipped on the next invocation"
        ),
        default=0,
        type=int,
    )
    parser.add_argument(
        "--mem_limit_gb",
        help=(
            "Abort a case with a MemoryError if resident memory exceeds this many GiB. "
            "Set it just below the point where the OOM killer fires so the failure is "
            "recorded instead of silent (0 disables)"
        ),
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--keep_going",
        help=(
            "Log and skip a case that raises instead of aborting the run. Failed IDs "
            "are written to <out>/failed_cids.txt for a targeted retry with --i"
        ),
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

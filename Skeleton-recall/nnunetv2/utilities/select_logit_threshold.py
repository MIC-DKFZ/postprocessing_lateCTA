#!/usr/bin/env python3
"""
Validation-set logit threshold selection for binary segmentation using clDice.

Like the Dice version, but optimises the centreline-Dice (clDice) metric of
Shit et al. (CVPR 2021), which rewards topological / connectivity agreement and
is the standard choice for tubular structures (vessels, airways, neurons).

For prediction P (= logit > t) and ground truth L, with S_P = skeleton(P) and
S_L = skeleton(L):
    Tprec = |S_P & L| / |S_P|        (predicted centreline inside GT)
    Tsens = |S_L & P| / |S_L|        (GT centreline inside prediction)
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

Because skeletonisation is nonlinear, the prediction is re-skeletonised at every
candidate threshold (the expensive part). Two optimisations keep it tractable:
  * the S_L side is threshold-independent, so Tsens is computed with searchsorted;
  * predictions are cropped to the structure bounding box before skeletonising.
Cases run in parallel with --workers.

Reference-layout handling (one-hot / permuted axes) is identical to the Dice
version: --ref-channel, --ref-channel-axis, --ref-transpose, plus an fg/bg
logit-separation sanity check.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required:  pip install nibabel")
try:
    from skimage.morphology import skeletonize
except ImportError:
    sys.exit("scikit-image is required:  pip install scikit-image")

from concurrent.futures import ProcessPoolExecutor


# --------------------------------------------------------------------------- #
# IO + reference alignment  (shared with the Dice version)
# --------------------------------------------------------------------------- #
def load_logit(path, dtype=np.float32):
    return np.squeeze(nib.load(path).get_fdata(dtype=np.float64).astype(dtype))


def _infer_channel_axis(shape, target_shape):
    tgt = sorted(target_shape)
    cands = [i for i in range(len(shape))
             if sorted(shape[:i] + shape[i + 1:]) == tgt]
    return min(cands, key=lambda i: shape[i]) if cands else None


def _find_permutation(src_shape, dst_shape):
    src = list(src_shape)
    perm, used = [], [False] * len(src)
    for d in dst_shape:
        for i, s in enumerate(src):
            if not used[i] and s == d:
                perm.append(i); used[i] = True; break
        else:
            return None, False
    return tuple(perm), (len(set(src)) != len(src))


def load_reference(path, target_shape, channel=-1,
                   channel_axis=None, transpose=None):
    arr = np.squeeze(np.asarray(nib.load(path).dataobj))
    notes = []
    if arr.ndim == len(target_shape) + 1:
        ax = channel_axis if channel_axis is not None \
            else _infer_channel_axis(arr.shape, target_shape)
        if ax is None:
            raise ValueError(f"cannot locate class axis in {arr.shape}")
        arr = np.take(arr, channel, axis=ax)
        notes.append(f"channel {channel} of axis {ax}")
    elif arr.ndim != len(target_shape):
        raise ValueError(f"ndim {arr.ndim} incompatible with {target_shape}")
    if transpose is not None:
        arr = np.transpose(arr, transpose); notes.append(f"transpose {transpose}")
    elif arr.shape != target_shape:
        perm, ambiguous = _find_permutation(arr.shape, target_shape)
        if perm is None:
            raise ValueError(f"{arr.shape} not a permutation of {target_shape}")
        arr = np.transpose(arr, perm)
        notes.append(f"auto-transpose {perm}"
                     + (" [AMBIGUOUS - verify!]" if ambiguous else ""))
    if arr.shape != target_shape:
        raise ValueError(f"aligned shape {arr.shape} != {target_shape}")
    return arr > 0, "; ".join(notes) if notes else "as-is"


def find_pairs(pred_dir, ref_dir, logit_suffix, ref_suffix):
    pairs = []
    for lp in sorted(glob.glob(os.path.join(pred_dir, "*" + logit_suffix))):
        cid = os.path.basename(lp)[: -len(logit_suffix)]
        rp = os.path.join(ref_dir, cid + ref_suffix)
        if os.path.exists(rp):
            pairs.append((cid, lp, rp))
        else:
            print(f"  [warn] no reference for '{cid}' - skipped")
    return pairs


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def bbox_slices(mask, margin, shape):
    if not mask.any():
        return None
    coords = np.array(np.nonzero(mask))
    lo = np.maximum(coords.min(axis=1) - margin, 0)
    hi = np.minimum(coords.max(axis=1) + 1 + margin, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def dice_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom == 0, 1.0, (2.0 * tp) / np.maximum(denom, 1))


# --------------------------------------------------------------------------- #
# Per-case worker
# --------------------------------------------------------------------------- #
def process_case(task):
    (cid, lp, rp, thresholds, channel, channel_axis, transpose, margin) = task
    try:
        logits = load_logit(lp)
        gt, info = load_reference(rp, logits.shape, channel=channel,
                                  channel_axis=channel_axis, transpose=transpose)
    except ValueError as e:
        return {"cid": cid, "error": str(e)}

    shape = np.array(logits.shape)
    gt_empty = not gt.any()
    T = thresholds.size

    # ---- Dice + prediction counts (cheap, via sorted logits) -------------- #
    pos = np.sort(logits[gt].ravel())
    neg = np.sort(logits[~gt].ravel())
    tp = pos.size - np.searchsorted(pos, thresholds, side="right")
    fp = neg.size - np.searchsorted(neg, thresholds, side="right")
    fn = pos.size - tp
    dice = dice_from_counts(tp, fp, fn)
    pred_count = tp + fp

    # ---- Tsens side: fixed GT skeleton, threshold-independent ------------- #
    tsens_num = np.zeros(T, np.int64)
    tsens_den = 0
    if not gt_empty:
        sl_sub = bbox_slices(gt, margin, shape)
        S_L_full = np.zeros(logits.shape, bool)
        S_L_full[sl_sub] = skeletonize(np.ascontiguousarray(gt[sl_sub]))
        skel_logits = np.sort(logits[S_L_full].ravel())
        tsens_den = skel_logits.size
        tsens_num = skel_logits.size - np.searchsorted(
            skel_logits, thresholds, side="right")

    # ---- Tprec side: skeletonise the prediction at each threshold --------- #
    tprec_num = np.zeros(T, np.int64)
    tprec_den = np.zeros(T, np.int64)
    for k, t in enumerate(thresholds):
        pred = logits > t
        if not pred.any():
            continue
        union = pred | gt
        sl = bbox_slices(union, margin, shape)
        S_P = skeletonize(np.ascontiguousarray(pred[sl]))
        tprec_den[k] = int(S_P.sum())
        if tprec_den[k]:
            tprec_num[k] = int((S_P & gt[sl]).sum())

    # ---- combine into clDice per threshold -------------------------------- #
    tprec = np.divide(tprec_num, tprec_den, out=np.zeros(T),
                      where=tprec_den > 0)
    tsens = (np.divide(tsens_num, tsens_den, out=np.zeros(T),
                       where=np.full(T, tsens_den > 0))
             if tsens_den else np.zeros(T))
    s = tprec + tsens
    cldice = np.where(s > 0, 2 * tprec * tsens / np.maximum(s, 1e-12), 0.0)
    if gt_empty:
        cldice = np.where(pred_count == 0, 1.0, 0.0)
    else:
        cldice = np.where(pred_count == 0, 0.0, cldice)

    sep = (float(logits[gt].mean() - logits[~gt].mean())
           if not gt_empty else float("nan"))

    return {"cid": cid, "info": info, "gt_empty": gt_empty, "sep": sep,
            "cldice": cldice, "dice": dice,
            "tprec_num": tprec_num, "tprec_den": tprec_den,
            "tsens_num": tsens_num, "tsens_den": int(tsens_den)}


# --------------------------------------------------------------------------- #
# Threshold grid
# --------------------------------------------------------------------------- #
def build_thresholds(pairs, n_thresholds, pct_lo, pct_hi,
                     tmin, tmax, sample_per_case, seed=0):
    if tmin is None or tmax is None:
        print("Pass 1/2: estimating logit range ...")
        rng = np.random.default_rng(seed)
        pool = []
        for cid, lp, _ in pairs:
            a = load_logit(lp).ravel()
            if a.size > sample_per_case:
                a = rng.choice(a, sample_per_case, replace=False)
            pool.append(a)
        lo, hi = np.percentile(np.concatenate(pool), [pct_lo, pct_hi])
        if tmin is not None: lo = tmin
        if tmax is not None: hi = tmax
    else:
        lo, hi = tmin, tmax
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -10.0, 10.0

    thr = np.unique(np.linspace(lo, hi, n_thresholds))
    print(f"  threshold grid: {thr.size} points in [{thr[0]:.3f}, {thr[-1]:.3f}]")
    return thr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--logit-suffix", default="_logit.nii.gz")
    ap.add_argument("--ref-suffix", default=".nii.gz")
    ap.add_argument("--n-thresholds", type=int, default=25,
                    help="clDice is expensive; keep this modest (default 25)")
    ap.add_argument("--pct", type=float, nargs=2, default=(0.5, 99.5),
                    metavar=("LO", "HI"))
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--agg", choices=["mean", "global"], default="mean",
                    help="mean per-case clDice, or micro-averaged (pooled counts)")
    ap.add_argument("--exclude-empty", action="store_true")
    ap.add_argument("--sample-per-case", type=int, default=200_000)
    ap.add_argument("--bbox-margin", type=int, default=2,
                    help="voxels of padding around the crop before skeletonising")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ref-channel", type=int, default=-1)
    ap.add_argument("--ref-channel-axis", type=int, default=None)
    ap.add_argument("--ref-transpose", type=str, default=None)
    ap.add_argument("--min-separation", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    transpose = (tuple(int(x) for x in args.ref_transpose.split(","))
                 if args.ref_transpose else None)

    pairs = find_pairs(args.pred_dir, args.ref_dir,
                       args.logit_suffix, args.ref_suffix)
    if not pairs:
        sys.exit("No matching (logit, reference) pairs found.")
    print(f"Found {len(pairs)} case(s).")

    thresholds = build_thresholds(pairs, args.n_thresholds, args.pct[0],
                                  args.pct[1], args.tmin, args.tmax,
                                  args.sample_per_case)

    tasks = [(cid, lp, rp, thresholds, args.ref_channel, args.ref_channel_axis,
              transpose, args.bbox_margin) for cid, lp, rp in pairs]

    print(f"Pass 2/2: clDice curves ({args.workers} worker(s)) ...")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(process_case, tasks))
    else:
        results = [process_case(t) for t in tasks]

    T = thresholds.size
    cl_rows, di_rows, case_ids = [], [], []
    g = {k: np.zeros(T, np.int64) for k in
         ("tprec_num", "tprec_den", "tsens_num")}
    g_tsens_den = 0
    n_empty = n_suspect = 0

    for r in results:
        if "error" in r:
            print(f"  [warn] '{r['cid']}': {r['error']} - skipped"); continue
        if r["gt_empty"]:
            n_empty += 1
            if args.exclude_empty:
                continue
        elif r["sep"] <= args.min_separation:
            n_suspect += 1
            print(f"  [ALIGN?] '{r['cid']}': fg-bg separation = {r['sep']:+.3f} "
                  f"- check --ref-channel/--ref-transpose. ({r['info']})")
        cl_rows.append(r["cldice"]); di_rows.append(r["dice"])
        case_ids.append(r["cid"])
        for k in ("tprec_num", "tprec_den", "tsens_num"):
            g[k] += r[k]
        g_tsens_den += r["tsens_den"]

    if not cl_rows:
        sys.exit("No usable cases after filtering.")

    cl = np.vstack(cl_rows)
    di = np.vstack(di_rows)
    mean_cl = cl.mean(0); std_cl = cl.std(0)
    mean_di = di.mean(0)

    # micro-averaged (global) clDice from pooled counts
    gt_tprec = np.divide(g["tprec_num"], g["tprec_den"], out=np.zeros(T),
                         where=g["tprec_den"] > 0)
    gt_tsens = (g["tsens_num"] / g_tsens_den if g_tsens_den else np.zeros(T))
    gs = gt_tprec + gt_tsens
    global_cl = np.where(gs > 0, 2 * gt_tprec * gt_tsens / np.maximum(gs, 1e-12), 0.0)

    curve = mean_cl if args.agg == "mean" else global_cl
    best_idx = int(np.argmax(curve))
    best_t = float(thresholds[best_idx])
    best_prob = 1.0 / (1.0 + np.exp(-best_t))
    zero_idx = int(np.argmin(np.abs(thresholds)))
    oracle = cl.max(1).mean()

    print("\n" + "=" * 62)
    print(f"Cases used             : {len(case_ids)}  ({n_empty} empty masks)")
    if n_suspect:
        print(f"Alignment warnings     : {n_suspect} - see [ALIGN?] above")
    print(f"Optimising             : {args.agg} clDice")
    print(f"Best logit threshold   : {best_t:.4f}   (prob = {best_prob:.4f})")
    print(f"  mean clDice          : {mean_cl[best_idx]:.4f} +/- {std_cl[best_idx]:.4f}")
    print(f"  global clDice        : {global_cl[best_idx]:.4f}")
    print(f"  mean Dice (same thr) : {mean_di[best_idx]:.4f}")
    print(f"clDice @ logit 0       : mean {mean_cl[zero_idx]:.4f}")
    print(f"Per-case oracle clDice : {oracle:.4f}  (upper bound)")
    di_best = int(np.argmax(mean_di))
    print(f"(For reference, Dice-optimal threshold = {thresholds[di_best]:.4f}, "
          f"clDice there = {mean_cl[di_best]:.4f})")
    print("=" * 62)

    curve_csv = os.path.join(args.out_dir, "cldice_threshold_curve.csv")
    with open(curve_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["logit_threshold", "probability", "mean_cldice",
                    "std_cldice", "global_cldice", "mean_dice"])
        for i, t in enumerate(thresholds):
            w.writerow([f"{t:.6f}", f"{1/(1+np.exp(-t)):.6f}",
                        f"{mean_cl[i]:.6f}", f"{std_cl[i]:.6f}",
                        f"{global_cl[i]:.6f}", f"{mean_di[i]:.6f}"])

    percase_csv = os.path.join(args.out_dir, "cldice_per_case_at_best.csv")
    with open(percase_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "cldice_at_best", "dice_at_best",
                    "case_optimal_threshold"])
        best_case_thr = thresholds[cl.argmax(1)]
        for cid, c_best, d_best, ct in zip(
                case_ids, cl[:, best_idx], di[:, best_idx], best_case_thr):
            w.writerow([cid, f"{c_best:.6f}", f"{d_best:.6f}", f"{ct:.6f}"])

    print(f"Wrote {curve_csv}")
    print(f"Wrote {percase_csv}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4.5))
            plt.plot(thresholds, mean_cl, marker="o", ms=3, label="mean clDice")
            plt.fill_between(thresholds, mean_cl - std_cl, mean_cl + std_cl, alpha=0.2)
            plt.plot(thresholds, global_cl, "--", label="global clDice")
            plt.plot(thresholds, mean_di, ":", color="grey", label="mean Dice")
            plt.axvline(best_t, color="k", ls=":", label=f"best = {best_t:.3f}")
            plt.xlabel("logit threshold"); plt.ylabel("score")
            plt.title("clDice vs. logit threshold (validation set)")
            plt.legend(); plt.tight_layout()
            png = os.path.join(args.out_dir, "cldice_threshold_curve.png")
            plt.savefig(png, dpi=130); print(f"Wrote {png}")
        except ImportError:
            print("  [warn] matplotlib not installed - skipping plot")


if __name__ == "__main__":
    main()
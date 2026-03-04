import argparse
import h5py
import numpy as np
import json
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


# ==========================================================
# Utilities
# ==========================================================


def load_h5(path):
    data = {}
    with h5py.File(path, "r") as f:
        for case_id in f.keys():
            grp = f[case_id]

            data[case_id] = {
                "pred_boxes": grp["pred_boxes"][:],
                "pred_scores": grp["pred_scores"][:],
                "pred_labels": grp["pred_labels"][:],
                "gt_labels": grp["gt_labels"][:],
                "pred_dists": grp["pred_dists"][:],
                "pred_times": grp["pred_times"][:],
                "pred_posterior": grp["pred_posterior"][:],
                "tp": grp["tp"][:],  # 1 if matched GT, 0 otherwise
            }

    return data


def compute_metrics(data, distance_thr=None, time_thr=None, remove_post=False):
    total_cases = len(data)
    detected_cases = 0
    total_fp = 0

    for case_id, case in data.items():
        is_tp = case["tp"]
        distances = case["pred_dists"]
        times = case["pred_times"]
        posterior = case["pred_posterior"]

        keep = np.ones_like(is_tp, dtype=bool)

        if distance_thr is not None:
            keep &= distances <= distance_thr

        if time_thr is not None:
            keep &= times <= time_thr  # <= as requested

        if remove_post:
            keep &= posterior == 0

        kept_tp = np.any((is_tp == 1) & keep)

        if kept_tp:
            detected_cases += 1

        fp = np.sum((is_tp == 0) & keep)
        total_fp += fp

    sensitivity = detected_cases / total_cases
    fp_per_scan = total_fp / total_cases

    return sensitivity, fp_per_scan


# ==========================================================
# Worker for parallelization
# ==========================================================


def evaluate_combo(combo, data):
    d_thr, t_thr, remove_post = combo

    sens, fp = compute_metrics(
        data,
        distance_thr=d_thr,
        time_thr=t_thr,
        remove_post=remove_post,
    )

    return {
        "distance_threshold": d_thr,
        "time_threshold": t_thr,
        "remove_posterior": remove_post,
        "sensitivity": sens,
        "fp_per_scan": fp,
    }


# ==========================================================
# Main
# ==========================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=cpu_count())

    parser.add_argument(
        "--distance_thresholds",
        type=float,
        nargs="+",
        default=[2, 4, 6, 8, 10],
    )

    parser.add_argument(
        "--time_thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.5, 1.0],
    )

    args = parser.parse_args()

    print("Loading H5 data...")
    data = load_h5(args.h5_file)

    # ------------------------------------------------------
    # Baseline (no filtering)
    # ------------------------------------------------------
    print("Computing baseline metrics...")
    baseline_sens, baseline_fp = compute_metrics(data)

    print(f"Baseline Sensitivity: {baseline_sens:.4f}")
    print(f"Baseline FP/scan:     {baseline_fp:.4f}")

    # ------------------------------------------------------
    # Hyperparameter combinations
    # ------------------------------------------------------
    combos = list(
        product(
            args.distance_thresholds,
            args.time_thresholds,
            [False, True],
        )
    )

    print(f"Evaluating {len(combos)} combinations with {args.num_workers} workers...")

    worker = partial(evaluate_combo, data=data)

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(worker, combos), total=len(combos)))

    # ------------------------------------------------------
    # Filter valid (sensitivity >= baseline)
    # ------------------------------------------------------
    valid = [r for r in results if r["sensitivity"] >= baseline_sens]

    if len(valid) == 0:
        print("No configuration keeps baseline sensitivity.")
        best = None
    else:
        best = min(valid, key=lambda r: r["fp_per_scan"])

    # ------------------------------------------------------
    # Save everything to JSON
    # ------------------------------------------------------
    output = {
        "baseline": {
            "sensitivity": float(baseline_sens),
            "fp_per_scan": float(baseline_fp),
        },
        "search_space": {
            "distance_thresholds": args.distance_thresholds,
            "time_thresholds": args.time_thresholds,
            "remove_posterior_options": [False, True],
        },
        "all_results": results,
        "optimal": best,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=4)

    print("\n===================================")
    print("Results saved to:", args.output_json)

    if best is not None:
        print("\nBEST CONFIG:")
        print(best)
    else:
        print("\nNo valid configuration found.")


if __name__ == "__main__":
    main()

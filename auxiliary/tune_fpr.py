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
                "gt_boxes": grp["gt_boxes"][:],
                "pred_dists": grp["pred_dists"][:],
                "pred_times": grp["pred_times"][:],
                "pred_posterior": grp["pred_posterior"][:],
                "tp": grp["tp"][:],  # 1 if matched GT, 0 otherwise
            }
    return data


def compute_metrics(data, distance_thr=None, time_thr=None, remove_post=False):
    """
    Compute case-level sensitivity, FP per scan, specificity, PPV, and NPV.
    """
    total_cases = len(data)
    detected_cases = 0  # TP cases
    total_fp = 0  # box-level FPs
    total_positives = 0  # cases with GT boxes
    tn_cases = 0  # TN cases

    predicted_positive_cases = 0
    predicted_negative_cases = 0

    for case_id, case in data.items():
        is_tp = case["tp"]
        distances = case["pred_dists"]
        times = case["pred_times"]
        posterior = case["pred_posterior"]
        gt_boxes = case["gt_boxes"]

        # Apply thresholds
        keep = np.ones_like(is_tp, dtype=bool)
        if distance_thr is not None:
            keep &= distances <= distance_thr
        if time_thr is not None:
            keep &= times <= time_thr
        if remove_post:
            keep &= posterior == 0

        has_tp_box = np.any((is_tp == 1) & keep)
        has_any_box = np.any(keep)

        # Case-level TP/TN
        if gt_boxes.shape[0] > 0:
            total_positives += 1
            if has_tp_box:
                detected_cases += 1
            if has_any_box:
                predicted_positive_cases += 1
            else:
                predicted_negative_cases += 1
        else:
            # negative case
            if keep.sum() == 0:
                tn_cases += 1
                predicted_negative_cases += 1
            else:
                predicted_positive_cases += 1

        # box-level FP
        total_fp += np.sum((is_tp == 0) & keep)

    # Metrics
    sensitivity = detected_cases / (total_positives + np.finfo(float).eps)
    n_negative_cases = total_cases - total_positives
    specificity = tn_cases / (n_negative_cases + np.finfo(float).eps)
    fp_per_scan = total_fp / total_cases
    ppv = detected_cases / (predicted_positive_cases + np.finfo(float).eps)
    npv = tn_cases / (predicted_negative_cases + np.finfo(float).eps)

    return sensitivity, fp_per_scan, specificity, ppv, npv


def bootstrap_metrics(
    data,
    distance_thr=None,
    time_thr=None,
    remove_post=False,
    n_iterations=1000,
    seed=42,
):
    """
    Case-level bootstrap for sensitivity, FP/scan, specificity, PPV, NPV.
    """
    rng = np.random.default_rng(seed)
    case_ids = list(data.keys())
    n_cases = len(case_ids)

    sens_values = []
    fp_values = []
    spec_values = []
    ppv_values = []
    npv_values = []

    for _ in range(n_iterations):
        sampled_ids = rng.choice(case_ids, size=n_cases, replace=True)
        sampled_data = {cid: data[cid] for cid in sampled_ids}

        sens, fp, spec, ppv, npv = compute_metrics(
            sampled_data,
            distance_thr=distance_thr,
            time_thr=time_thr,
            remove_post=remove_post,
        )

        sens_values.append(sens)
        fp_values.append(fp)
        spec_values.append(spec)
        ppv_values.append(ppv)
        npv_values.append(npv)

    def summarize(arr):
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)),
            "p2_5": float(np.percentile(arr, 2.5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p97_5": float(np.percentile(arr, 97.5)),
        }

    return {
        "sensitivity": summarize(sens_values),
        "fp_per_scan": summarize(fp_values),
        "specificity": summarize(spec_values),
        "ppv": summarize(ppv_values),
        "npv": summarize(npv_values),
    }


# ==========================================================
# Worker for parallelization
# ==========================================================


def evaluate_combo(combo, data):
    d_thr, t_thr, remove_post = combo
    sens, fp, spec, ppv, npv = compute_metrics(
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
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
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
        default=list(range(2, 41, 2)),
    )
    parser.add_argument(
        "--time_thresholds",
        type=float,
        nargs="+",
        default=np.round(np.arange(0.33, 0.51 + 1e-9, 0.01), 2).tolist(),
    )
    args = parser.parse_args()

    print("Loading H5 data...")
    data = load_h5(args.h5_file)

    # ------------------------------------------------------
    # Baseline (no filtering)
    # ------------------------------------------------------
    print("Computing baseline metrics...")
    baseline_sens, baseline_fp, baseline_spec, baseline_ppv, baseline_npv = (
        compute_metrics(data)
    )
    print(f"Baseline Sensitivity: {baseline_sens:.4f}")
    print(f"Baseline FP/scan:     {baseline_fp:.4f}")
    print(f"Baseline Specificity: {baseline_spec:.4f}")
    print(f"Baseline PPV:         {baseline_ppv:.4f}")
    print(f"Baseline NPV:         {baseline_npv:.4f}")

    baseline_bootstrap = bootstrap_metrics(
        data, distance_thr=None, time_thr=None, remove_post=False, n_iterations=1000
    )

    # ------------------------------------------------------
    # Hyperparameter combinations
    # ------------------------------------------------------
    combos = list(
        product(args.distance_thresholds, args.time_thresholds, [False, True])
    )
    print(f"Evaluating {len(combos)} combinations with {args.num_workers} workers...")
    worker = partial(evaluate_combo, data=data)

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(worker, combos), total=len(combos)))

    # ------------------------------------------------------
    # Filter valid (sensitivity >= baseline)
    # ------------------------------------------------------
    best = max(results, key=lambda r: (round(r["sensitivity"], 6), -r["fp_per_scan"]))
    selection_mode = "max_sensitivity_then_min_fp"

    # ------------------------------------------------------
    # Bootstrap best config
    # ------------------------------------------------------
    if best is not None:
        print("\nBEST CONFIG:")
        print(best)
        best_bootstrap = bootstrap_metrics(
            data,
            distance_thr=best["distance_threshold"],
            time_thr=best["time_threshold"],
            remove_post=best["remove_posterior"],
            n_iterations=1000,
        )
    else:
        best_bootstrap = None

    # ------------------------------------------------------
    # Save everything to JSON
    # ------------------------------------------------------
    output = {
        "baseline": {
            "sensitivity": float(baseline_sens),
            "fp_per_scan": float(baseline_fp),
            "specificity": float(baseline_spec),
            "ppv": float(baseline_ppv),
            "npv": float(baseline_npv),
        },
        "baseline_bootstrap_1000": baseline_bootstrap,
        "search_space": {
            "distance_thresholds": args.distance_thresholds,
            "time_thresholds": args.time_thresholds,
            "remove_posterior_options": [False, True],
        },
        "all_results": results,
        "optimal": best,
        "best_bootstrap_1000": best_bootstrap,
        "selection_mode": selection_mode,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=4)

    print("\n===================================")
    print("Results saved to:", args.output_json)


if __name__ == "__main__":
    main()

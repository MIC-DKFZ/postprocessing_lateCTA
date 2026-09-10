# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import argparse
import numpy as np
import time
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json


def main(args):
    folder_file = args.f
    outfile = args.o

    assert os.path.exists(folder_file) and folder_file.endswith(
        ".json"
    ), f"Folder file '{folder_file}' does not exist or is not .json"
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(
        ".json"
    ), f"Parent output directory '{os.path.dirname(outfile)}' does not exist or output file is not .json"

    # Load folders
    folder_info = load_json(folder_file)
    methods = list(folder_info.keys())
    out = {}

    metric_mapping = {
        "box_sens": "Occlusion sensitivity",
        "fp_scan": "False occlusions per scan",
        "auc_case": "AUROC",
        "sens95_case": "Sensitivity @95 specificity",
        "spec95_case": "Specificity @95 specificity",
        "auprc_case": "AUPRC",
    }
    all_metrics = np.array(list(metric_mapping.keys()), dtype=str)
    metric_names = np.array(list(metric_mapping.values()), dtype=str)

    for method in methods:
        method_info = {}
        # Extract FROC information
        froc_file = os.path.join(folder_info[method]["froc"], "results_boxes_boot.json")
        assert os.path.exists(froc_file), f"FROC file '{froc_file}' does not exist"

        froc_info = np.array(load_json(froc_file)["FROCwp_IoU_0.10"])
        method_info["FROC"] = {
            "median": 100 * np.nanmedian(froc_info),
            "mean": 100 * np.nanmean(froc_info),
            "std": 100 * np.nanstd(froc_info),
            "inf": 100 * np.nanpercentile(froc_info, 25),
            "sup": 100 * np.nanpercentile(froc_info, 75),
            "min": 100 * np.nanmin(froc_info),
            "max": 100 * np.nanmax(froc_info),
            "p2_5": 100 * np.nanpercentile(froc_info, 2.5),
            "p97_5": 100 * np.nanpercentile(froc_info, 97.5),
        }

        # Extract remaining metric information
        metric_file = folder_info[method]["sens_file"]
        assert os.path.exists(
            metric_file
        ), f"Metric file '{metric_file}' does not exist"
        metric_info = load_json(metric_file)

        avail_metrics = np.array(list(metric_info.keys()))

        metrics, ind, _ = np.intersect1d(
            all_metrics, avail_metrics, return_indices=True
        )
        metric_names_chosen = metric_names[ind]

        for metric, metric_name in zip(metrics, metric_names_chosen):
            if metric == "fp_scan":
                factor = 1.0
            else:
                factor = 100.0  # Express metric in %

            info = metric_info[metric]
            try:
                method_info[metric_name] = {
                    "median": factor * info["median"],
                    "mean": factor * info["mean"],
                    "std": factor * info["std"],
                    "p2_5": factor * info["p2_5"],
                    "p97_5": factor * info["p97_5"],
                    "min": factor * info["min"],
                    "max": factor * info["max"],
                    "inf": factor * info["p25"],
                    "sup": factor * info["p75"],
                }
            except:
                method_info[metric_name] = {
                    "median": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "p2_5": 0.0,
                    "p97_5": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "inf": 0.0,
                    "sup": 0.0,
                }

        out[method] = method_info

    save_json(out, outfile)


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate json file with data to be plotted for manuscript"
    )
    parser.add_argument(
        "--f", help="File with folder information", required=True, type=str
    )
    parser.add_argument("--o", help="Output file", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

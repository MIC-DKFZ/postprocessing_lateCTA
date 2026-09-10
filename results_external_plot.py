# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os, sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse
from batchgenerators.utilities.file_and_folder_operations import load_json
import time


def build_stats(method, metrics, data):
    stats = []
    for m in metrics:
        d = data[method][m]
        stats.append(
            {
                "med": d["median"],
                "q1": d["inf"],
                "q3": d["sup"],
                "whislo": d["min"],
                "whishi": d["max"],
                "fliers": [],
            }
        )
    return stats


def build_means(method, metrics, data):
    return [data[method][m]["mean"] for m in metrics]


def build_medians(method, metrics, data):
    return [data[method][m]["median"] for m in metrics]


def main(args):
    infile = args.i
    outfile = args.o

    assert os.path.exists(infile) and infile.endswith(".json")
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(".png")

    data = load_json(infile)

    methods = [
        "Baseline",
        "Time-vessel map (extra channel)",
        "False occlusion removal",
    ]

    metrics = [
        "FROC",
        "Occlusion sensitivity",
        "AUROC",
        "Sensitivity @90 specificity",
        "Specificity @90 specificity",
    ]

    metric_labels = [
        "FROC",
        "Occlusion sensitivity",
        "AUROC",
        "Sensitivity @90\nspecificity",
        "Specificity @90\nsensitivity",
    ]

    # spacing between metric groups
    spacing = 2.0

    # auto-centered positions for any number of methods
    n_methods = len(methods)
    offsets = np.linspace(-0.4, 0.4, n_methods)
    positions = [np.arange(len(metrics)) * spacing + off for off in offsets]

    # figure for 150x60 mm
    fig, ax = plt.subplots(figsize=(11.69, 3.5))

    colors = [
        "#4A90E2",  # Baseline — bright blue
        "#82B1FF",  # Time vessel map — lighter blue
        "#1B3A6F",  # False occlusion removal — dark blue
    ]

    boxplots = []

    # -------------------------
    # Main plotting loop
    # -------------------------
    for i, method in enumerate(methods):
        stats = build_stats(method, metrics, data)
        means = build_means(method, metrics, data)
        medians = build_medians(method, metrics, data)

        b = ax.bxp(
            stats,
            positions=positions[i],
            widths=0.3,
            showfliers=False,
            patch_artist=True,
        )

        # box colors
        for box in b["boxes"]:
            box.set_facecolor(colors[i])

        # median styling
        for median in b["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        # whiskers + caps
        for whisker in b["whiskers"]:
            whisker.set_linewidth(1.2)
        for cap in b["caps"]:
            cap.set_linewidth(1.2)

        # annotate medians
        offset = 0.02
        for x, stat, median_val in zip(positions[i], stats, medians):
            ax.text(
                x,
                stat["whishi"] + offset,
                f"{median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

        # mean markers
        for x, mean in zip(positions[i], means):
            ax.scatter(x, mean, color="orange", marker="x", s=30, zorder=3)

        boxplots.append(b)

    # -------------------------
    # Axes styling
    # -------------------------
    ax.set_xticks(np.arange(len(metrics)) * spacing)
    ax.set_xticklabels(metric_labels, fontsize=12)

    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["30", "60", "90"], fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------
    # Legend
    # -------------------------
    handles = [b["boxes"][0] for b in boxplots]

    legend_labels = []
    for m in methods:
        f = data[m]["False occlusions per scan"]
        label = f"{m}\n" f"{f['median']:.2f} (95% CI: {f['p2_5']:.2f}–{f['p97_5']:.2f})"
        legend_labels.append(label)

    mean_handle = Line2D(
        [0], [0], color="orange", marker="x", linestyle="None", markersize=6
    )
    handles.append(mean_handle)
    legend_labels.append("Mean")

    leg = ax.legend(
        handles,
        legend_labels,
        title="\n\nFalse occlusions per scan (N = 354)",
        fontsize=12,
        title_fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(handles),
        frameon=False,
    )

    for text in leg.get_texts():
        text.set_ha("center")

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.40)

    plt.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Obtain result plot for external cohort"
    )
    parser.add_argument("--i", help="Input file", type=str)
    parser.add_argument("--o", help="Output file", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

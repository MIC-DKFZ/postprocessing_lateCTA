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
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(".svg")

    data = load_json(infile)

    methods = ["Baseline", "False occlusion removal"]

    metrics = [
        "FROC",
        "Occlusion sensitivity",
        "AUROC",
        "Sensitivity @90 specificity",
        "Specificity @90 specificity",
    ]

    metric_labels = [
        "FROC",
        "Occlusion sens",
        "AUROC",
        "Sens @90\nspec",
        "Spec @90\nsens",
    ]

    stats_baseline = build_stats(methods[0], metrics, data)
    stats_removal = build_stats(methods[1], metrics, data)

    means_baseline = build_means(methods[0], metrics, data)
    medians_baseline = build_medians(methods[0], metrics, data)
    means_removal = build_means(methods[1], metrics, data)
    medians_removal = build_medians(methods[1], metrics, data)

    # -------------------
    # Figure and positions
    # -------------------
    fig_width_mm = 127
    fig_height_mm = 30
    fig, ax = plt.subplots(figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4))

    # spacing between boxplot groups
    spacing = 1.0
    positions1 = np.arange(len(metrics)) * spacing - 0.2
    positions2 = np.arange(len(metrics)) * spacing + 0.2

    bright_blue = "#4A90E2"
    dark_blue = "#1B3A6F"

    # -------------------
    # Boxplots
    # -------------------
    b1 = ax.bxp(
        stats_baseline,
        positions=positions1,
        widths=0.3,
        showfliers=False,
        patch_artist=True,
    )

    b2 = ax.bxp(
        stats_removal,
        positions=positions2,
        widths=0.3,
        showfliers=False,
        patch_artist=True,
    )

    # -------------------
    # Annotate median values
    # -------------------
    offset = 0.01
    for x, stat, median in zip(positions1, stats_baseline, medians_baseline):
        ax.text(
            x,
            stat["whishi"] + offset,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    for x, stat, median in zip(positions2, stats_removal, medians_removal):
        ax.text(
            x,
            stat["whishi"] + offset,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    # -------------------
    # Box colors
    # -------------------
    for box in b1["boxes"]:
        box.set_facecolor(bright_blue)

    for box in b2["boxes"]:
        box.set_facecolor(dark_blue)

    # Median line styling
    for median in b1["medians"] + b2["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    # Whiskers and caps
    for whisker in b1["whiskers"] + b2["whiskers"]:
        whisker.set_linewidth(1.0)

    for cap in b1["caps"] + b2["caps"]:
        cap.set_linewidth(1.0)

    # -------------------
    # Axes and labels
    # -------------------
    ax.set_xticks(np.arange(len(metrics)) * spacing)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks([])  # hide y-axis labels
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------
    # Legend
    # -------------------
    legend_labels = []
    for m in methods:
        f = data[m]["False occlusions per scan"]
        label = f"{m}\n{f['median']:.2f} [{f['inf']:.2f}-{f['sup']:.2f}]"
        legend_labels.append(label)

    handles = [b1["boxes"][0], b2["boxes"][0]]

    leg = ax.legend(
        handles,
        legend_labels,
        title="False occlusions per scan",
        fontsize=9,
        title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=len(methods),
        frameon=False,
    )

    for text in leg.get_texts():
        text.set_fontweight("bold")
        text.set_ha("center")

    # -------------------
    # Layout adjustments
    # -------------------
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(bottom=0.35)

    plt.savefig(outfile, format="svg", bbox_inches="tight", transparent=True, dpi=600)


def get_args():
    parser = argparse.ArgumentParser(description="Obtain plot for Graphical Abstract")
    parser.add_argument("--i", help="Input file", type=str)
    parser.add_argument("--o", help="Output file", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1:.2f} sec")

# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse
from batchgenerators.utilities.file_and_folder_operations import load_json
import time


def get_common_metrics(data, methods):
    """Only keep metrics present in ALL methods"""
    common = set(data[methods[0]].keys())
    for m in methods[1:]:
        common &= set(data[m].keys())
    return list(common)


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


def main(args):
    data = load_json(args.i)

    methods = [
        m
        for m in [
            "R1",
            "R2",
            "R3",
            "R1 + R2",
            "R1 + R3",
            "R2 + R3",
            "R1 + R2 + R3",
        ]
        if m in data
    ]

    # -------------------------
    # Only safe metrics
    # -------------------------
    metrics = ["FROC", "Occlusion sensitivity"]

    metric_labels = [m.replace(" ", "\n") for m in metrics]

    # -------------------------
    # Build data
    # -------------------------
    all_stats = [build_stats(m, metrics, data) for m in methods]
    all_means = [build_means(m, metrics, data) for m in methods]

    # -------------------------
    # Layout
    # -------------------------
    spacing = 2.5
    base_positions = np.arange(len(metrics)) * spacing

    n_methods = len(methods)
    offsets = np.linspace(-0.6, 0.6, n_methods)

    fig, ax = plt.subplots(figsize=(11.69, 3.5))  # A4 width

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_methods))
    handles = []

    # -------------------------
    # Plot
    # -------------------------
    for i, (method, stats, means) in enumerate(zip(methods, all_stats, all_means)):
        positions = base_positions + offsets[i]

        b = ax.bxp(
            stats,
            positions=positions,
            widths=0.15,
            showfliers=False,
            patch_artist=True,
        )

        # color
        for box in b["boxes"]:
            box.set_facecolor(colors[i])

        # medians
        for median in b["medians"]:
            median.set_color("black")
            median.set_linewidth(1.3)

        # mean crosses
        for x, mean in zip(positions, means):
            ax.scatter(x, mean, color="black", marker="x", s=30, zorder=3)

        handles.append(b["boxes"][0])

    # -------------------------
    # Axes
    # -------------------------
    ax.set_xticks(base_positions)
    ax.set_xticklabels(metric_labels, fontsize=12)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------
    # Legend (3 rows)
    # -------------------------
    legend_labels = []
    for m in methods:
        f = data[m]["False occlusions per scan"]
        label = f"{m}\n{f['median']:.2f} [{f['inf']:.2f}–{f['sup']:.2f}]"
        legend_labels.append(label)

    mean_handle = Line2D(
        [0], [0], color="black", marker="x", linestyle="None", markersize=6
    )

    handles.append(mean_handle)
    legend_labels.append("Mean")

    # FORCE 3 rows
    ncol = int(np.ceil(len(handles) / 3))

    leg = ax.legend(
        handles,
        legend_labels,
        title="False occlusions / scan",
        fontsize=10,
        title_fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=ncol,
        frameon=False,
    )

    for text in leg.get_texts():
        text.set_ha("center")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    plt.savefig(args.o, dpi=600, bbox_inches="tight")
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)
    parser.add_argument("--o", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time elapsed: {time.time()-t1:.2f}s")

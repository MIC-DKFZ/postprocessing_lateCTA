import os, sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse
from batchgenerators.utilities.file_and_folder_operations import load_json
import time


# -------------------------
# Helpers
# -------------------------
def build_stats(method, metric, data):
    d = data[method][metric]
    return {
        "med": d["median"],
        "q1": d["inf"],
        "q3": d["sup"],
        "whislo": d["min"],
        "whishi": d["max"],
        "fliers": [],
    }


def get_mean(method, metric, data):
    return data[method][metric]["mean"]


def get_median(method, metric, data):
    return data[method][metric]["median"]


def main(args):
    infile = args.i
    outfile = args.o

    assert os.path.exists(infile) and infile.endswith(".json")
    assert os.path.exists(os.path.dirname(outfile)) and outfile.endswith(".png")

    data = load_json(infile)

    # methods = [
    #    "Baseline",
    #    "Baseline + post-processing correction",
    #    "Late-phase retraining",
    #    "Late-phase retraining + post-processing correction",
    # ]
    methods = [
        "Baseline",
        "Time-vessel map (extra channel)",
        "Baseline + post-processing correction",
    ]
    # methods = list(data.keys())

    metrics = ["FROC", "Occlusion sensitivity"]

    # -------------------------
    # Layout
    # -------------------------
    spacing = 1.6
    n_methods = len(methods)

    base_positions = np.arange(len(metrics)) * spacing
    offsets = np.linspace(-0.35, 0.35, n_methods)

    # 150x60 mm
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    colors = [
        "#4A90E2",  # bright blue
        "#1B3A6F",  # dark blue
        "#7EC8E3",  # light blue
        "#A6D4FA",  # very light blue
    ]

    # -------------------------
    # Plot
    # -------------------------
    all_handles = []

    for i, method in enumerate(methods):
        positions = base_positions + offsets[i]

        stats = [build_stats(method, m, data) for m in metrics]
        means = [get_mean(method, m, data) for m in metrics]
        medians = [get_median(method, m, data) for m in metrics]

        b = ax.bxp(
            stats,
            positions=positions,
            widths=0.25,
            showfliers=False,
            patch_artist=True,
        )

        # color
        for box in b["boxes"]:
            box.set_facecolor(colors[i])

        # styling
        for median in b["medians"]:
            median.set_color("black")
            median.set_linewidth(1.3)

        for whisker in b["whiskers"]:
            whisker.set_linewidth(1.1)

        for cap in b["caps"]:
            cap.set_linewidth(1.1)

        # mean annotations
        for x, stat, median in zip(positions, stats, medians):
            ax.text(
                x,
                stat["whishi"] + 0.02,
                f"{round(median)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        for x, mean in zip(positions, means):
            ax.scatter(
                x,
                mean,
                marker="x",
                s=30,  # slightly larger for visibility
                color="orange",
                linewidths=1.5,
                zorder=3,
            )

        all_handles.append(b["boxes"][0])

    # -------------------------
    # Axes
    # -------------------------
    ax.set_xticks(base_positions)
    ax.set_xticklabels(["FROC", "Occlusion sensitivity"], fontsize=12)

    # ax.set_ylim(0.1, 1.05)
    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["30", "60", "90"], fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------
    # Legend (bottom with title)
    # -------------------------
    legend_labels = []
    for m in methods:
        f = data[m]["False occlusions per scan"]
        label = f"{m}\n" f"{f['median']:.1f} (95% CI: {f['p2_5']:.1f}–{f['p97_5']:.1f})"
        legend_labels.append(label)

    mean_handle = Line2D(
        [0],
        [0],
        marker="x",
        color="orange",
        linestyle="None",
        markersize=7,
        markeredgewidth=1.5,
    )
    handles = all_handles + [mean_handle]
    labels = legend_labels + ["Mean"]

    leg = ax.legend(
        handles,
        labels,
        title="Incorrect occlusion detections per scan (N = 32)\n",
        fontsize=11,
        title_fontsize=11,
        loc="center left",  # anchor legend from its left side
        bbox_to_anchor=(1.02, 0.5),  # push it to the right of the axes
        frameon=False,
    )

    for text in leg.get_texts():
        text.set_ha("left")

    leg._legend_box.sep = 6

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(right=0.75)

    plt.savefig(outfile, format="png", bbox_inches="tight", transparent=True, dpi=600)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="Obtain results plot")
    parser.add_argument("--i", help="Input file", type=str)
    parser.add_argument("--o", help="Output file", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

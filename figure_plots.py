import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import argparse
import time
import os, sys

plt.rcParams["font.size"] = 16  # default text
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.titlesize"] = 16  # title
plt.rcParams["axes.labelsize"] = 16  # axis labels
plt.rcParams["xtick.labelsize"] = 16  # x tick labels
plt.rcParams["ytick.labelsize"] = 16  # y tick labels
plt.rcParams["legend.fontsize"] = 14  # legend


def plot_metrics_boxplots(data, title: str = "", outfile: str = ""):
    """
    data structure:
    {
        method_name : {
            metric_name : {min, inf, median, sup, max, mean},
            ...,
            "FP": value
        },
        ...
    }
    """

    methods = list(data.keys())
    metrics = [m for m in data[methods[0]].keys() if m != "False occlusions per scan"]

    n_methods = len(methods)
    n_metrics = len(metrics)

    # Colors
    cmap = cm.get_cmap("tab10", n_methods)
    colors = [cmap(i) for i in range(n_methods)]

    # Prepare data
    box_data = {metric: [] for metric in metrics}
    for method in methods:
        for metric in metrics:
            d = data[method][metric]
            box_data[metric].append(
                [d["min"], d["inf"], d["median"], d["sup"], d["max"]]
            )

    # Figure
    fig, ax = plt.subplots(
        figsize=(10, 8)
    )  # (14,10) for external cohort // (10,8) for development cohort

    metric_spacing = 0.3  # 1.5 for external cohort, 0.3 for development cohort
    positions = np.arange(n_metrics) * metric_spacing
    width = 0.2 / n_methods  # 0.8 for external cohort, 0.1 for development cohort

    ### IMPORTANT: define xticklabels ONCE, not inside the loop
    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace(" ", "\n") for m in metrics])

    # Plot per method
    for i, method in enumerate(methods):
        stats = [box_data[m][i] for m in metrics]

        box = ax.bxp(
            [dict(med=s[2], q1=s[1], q3=s[3], whislo=s[0], whishi=s[4]) for s in stats],
            positions=positions + (i - n_methods / 2) * width + width / 2,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="orange", linewidth=2),
            whiskerprops=dict(color="black", linewidth=2),
            capprops=dict(color="black", linewidth=2),
            manage_ticks=False,
        )

        # Color boxes
        for patch in box["boxes"]:
            patch.set_facecolor(colors[i])
        # for item in box["whiskers"] + box["caps"]:
        # item.set_color(colors[i])

        # Add median + mean markers
        for j, metric in enumerate(metrics):
            s = stats[j]
            median_val = s[2]
            mean_val = data[method][metric]["mean"]

            x_pos = positions[j] + (i - n_methods / 2) * width + width / 2

            pos = median_val + 0.001
            # if metric == "AUROC" or metric == "Case specificity":
            if metric == "Occlusion precision":
                pos = s[-1] + 0.001

            ax.text(
                x_pos,
                pos,
                f"{median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black",
                fontweight="bold",
            )

            ax.plot(
                x_pos,
                mean_val,
                marker="X",
                markersize=6,
                color="black",
                label="_mean",
                zorder=5,
            )

    # Labels
    # ax.set_ylabel("Metric Value")
    ax.set_title(title)

    # Legend
    method_handles = [Line2D([0], [0], color=colors[i], lw=5) for i in range(n_methods)]
    # method_labels = [
    #    f"{method}\n(FP per image={data[method]['FP per image']:.2f})"
    #    for method in methods
    # ]

    method_labels = [
        (
            f"{method}\n(False occlusions\nper scan={data[method]['False occlusions per scan']:.2f})\n"
            if method == "Baseline"
            else f"False occlusion\nremoval\n(False occlusions\nper scan={data[method]['False occlusions per scan']:.2f})\n"
        )
        for method in methods
    ]

    mean_handle = Line2D(
        [0],
        [0],
        marker="X",
        color="black",
        markerfacecolor="black",
        markersize=6,
        linestyle="None",
    )

    ax.legend(
        handles=method_handles + [mean_handle],
        labels=method_labels + ["Mean (X)"],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        handlelength=1,
        handleheight=0.6,
        fontsize=12,
        prop={"family": "monospace"},
        alignment="center",
    )

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def main(args):
    file = args.f
    title = args.t
    outfile = args.o

    assert os.path.exists(file), f"Data file for plotting '{file}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfile)
    ), f"Output parent directory '{os.path.dirname(outfile)}' does not exist"

    data = load_json(file)
    plot_metrics_boxplots(data, title=title, outfile=outfile)


def get_args():
    # Plot results for Amsterdam's manuscript
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", help="Data plot file", required=True, type=str)
    parser.add_argument("--t", help="Title", default="", type=str)
    parser.add_argument("--o", help="Output file", required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

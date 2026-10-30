"""Figure 5: leading-model sensitivity to the training/test split."""

import os

from common import METRICS, save

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, StrMethodFormatter
import numpy as np
import pandas as pd


SPLITS = ["50/50", "60/40", "70/30", "80/20"]
MODEL_SERIES = [
    (
        "ExtraTrees",
        "Extremely randomised trees",
        "#595959",
        "s",
    ),
    (
        "GradientBoosting",
        "Gradient boosting",
        "#D55E00",
        "o",
    ),
    (
        "LightGBM",
        "LightGBM",
        "#0072B2",
        "^",
    ),
    (
        "NN_PINN_HARD",
        "Hard-constrained PINN",
        "#009E73",
        "v",
    ),
    (
        "XGBoost",
        "XGBoost",
        "#8E63CE",
        "D",
    ),
]

FIGURE_WIDTH = 11.0
FIGURE_HEIGHT = 8.0
AXIS_LABEL_SIZE = 36
TICK_LABEL_SIZE = 25


def main():
    if not os.path.exists(METRICS):
        print("skipped: missing", METRICS)
        return

    metrics = pd.read_csv(METRICS)
    positions = np.arange(len(SPLITS))
    fig, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig._wag_layout_pad = 1.0

    plotted = 0
    plotted_values = []
    for model, label, color, marker in MODEL_SERIES:
        values = (
            metrics.loc[metrics["model"] == model]
            .groupby("split")["R2"]
            .max()
            .reindex(SPLITS)
        )
        if values.isna().any():
            print(f"skipped series with incomplete split data: {model}")
            continue

        series_values = values.to_numpy()
        axis.plot(
            positions,
            series_values,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        plotted_values.extend(series_values)
        plotted += 1

    if not plotted:
        plt.close(fig)
        print("skipped: no complete split-sensitivity series found")
        return

    axis.set_xlim(positions[0], positions[-1])
    axis.set_ylim(min(plotted_values), max(plotted_values))

    axis.set_xticks(positions)
    axis.set_xticklabels(SPLITS)
    axis.yaxis.set_major_locator(LinearLocator(7))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    axis.set_xlabel("Training/test split", labelpad=14)
    axis.set_ylabel("Held-out R²", labelpad=14)
    axis._wag_axis_labelsize = AXIS_LABEL_SIZE
    axis._wag_tick_labelsize = TICK_LABEL_SIZE
    axis.xaxis._wag_show_ticks = True
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=8)
    axis.set_facecolor("#FBFBFB")

    axis.legend(
        loc="upper right",
        ncol=1,
        frameon=False,
        labelspacing=0.55,
        handlelength=2.2,
        borderaxespad=0.7,
    )

    save(fig, "Fig5_split_sensitivity")


if __name__ == "__main__":
    main()

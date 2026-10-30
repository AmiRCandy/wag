"""Figure 4: effect of physics guidance on model accuracy."""

import os

from common import CVMETRICS, save

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "LinearRegression": "Linear regression",
    "Ridge": "Ridge regression",
    "Lasso": "Lasso regression",
    "ElasticNet": "Elastic net",
    "KNN": "k-nearest neighbours",
    "SVR_rbf": "RBF support vector regression",
    "RandomForest": "Random forest",
    "ExtraTrees": "Extremely randomised trees",
    "GradientBoosting": "Gradient boosting",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "NeuralNet": "Neural network",
}

STANDARD_COLOR = "#595959"
PHYSICS_COLOR = "#00796B"
FIGURE_WIDTH = 16.0
AXIS_LABEL_SIZE = 36
TICK_LABEL_SIZE = 25
VALUE_LABEL_SIZE = 22


def main():
    if not os.path.exists(CVMETRICS):
        print("skipped: missing", CVMETRICS)
        return

    metrics = pd.read_csv(CVMETRICS).set_index("model")
    bases = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "KNN",
        "SVR_rbf",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
    ]
    rows = [
        (
            base,
            float(metrics.loc[base, "R2"]) * 100.0,
            float(metrics.loc[base + "_PHYS", "R2"]) * 100.0,
        )
        for base in bases
        if base in metrics.index and base + "_PHYS" in metrics.index
    ]
    if "NN_MLP(1000)" in metrics.index and "NN_PINN_HARD(1000)" in metrics.index:
        rows.append(
            (
                "NeuralNet",
                float(metrics.loc["NN_MLP(1000)", "R2"]) * 100.0,
                float(metrics.loc["NN_PINN_HARD(1000)", "R2"]) * 100.0,
            )
        )
    if not rows:
        print("skipped: no standard/physics-guided model pairs found")
        return

    rows.sort(key=lambda row: max(row[1], row[2]))
    labels = [MODEL_LABELS.get(row[0], row[0]) for row in rows]
    standard = np.asarray([row[1] for row in rows])
    physics = np.asarray([row[2] for row in rows])
    positions = np.arange(len(labels))
    bar_height = 0.34

    figure_height = 0.72 * len(rows) + 2.0
    fig, axis = plt.subplots(figsize=(FIGURE_WIDTH, figure_height))
    fig._wag_layout_pad = 1.7

    axis.barh(
        positions - bar_height / 2,
        standard,
        height=bar_height,
        color=STANDARD_COLOR,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
    )
    axis.barh(
        positions + bar_height / 2,
        physics,
        height=bar_height,
        color=PHYSICS_COLOR,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.axvline(0, color="black", linewidth=1.2)
    axis.axvline(100, color="none", linewidth=0)
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_xlabel("Five-fold cross-validated R² (%)", labelpad=14)
    axis._wag_axis_labelsize = AXIS_LABEL_SIZE
    axis._wag_tick_labelsize = TICK_LABEL_SIZE
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=10)
    axis.set_facecolor("#FBFBFB")

    for index, (standard_value, physics_value) in enumerate(
        zip(standard, physics)
    ):
        standard_label = axis.text(
            standard_value - 1.2,
            index - bar_height / 2,
            f"{standard_value:.1f}",
            ha="right",
            va="center",
            color="white",
        )
        standard_label._wag_fontsize = VALUE_LABEL_SIZE

        physics_label = axis.text(
            physics_value - 1.2,
            index + bar_height / 2,
            f"{physics_value:.1f}",
            ha="right",
            va="center",
            color="white",
        )
        physics_label._wag_fontsize = VALUE_LABEL_SIZE

    axis.legend(
        handles=[
            Patch(facecolor=STANDARD_COLOR, label="Standard model"),
            Patch(facecolor=PHYSICS_COLOR, label="Physics-guided model"),
        ],
        loc="lower right",
        ncol=1,
        frameon=False,
        labelspacing=0.7,
        handlelength=1.8,
        borderaxespad=0.8,
    )

    save(fig, "Fig4_physics_effect")


if __name__ == "__main__":
    main()

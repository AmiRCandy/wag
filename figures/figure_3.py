"""Figure 3: five-fold cross-validated R-squared ranking."""

import os

from common import CVMETRICS, save

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "XGBoost": "XGBoost",
    "NN_PINN_HARD(1000)": "Hard-constrained PINN (1,000 epochs)",
    "LightGBM": "LightGBM",
    "GradientBoosting": "Gradient boosting",
    "ExtraTrees": "Extremely randomised trees",
    "XGBoost_PHYS": "Physics-guided XGBoost",
    "GradientBoosting_PHYS": "Physics-guided gradient boosting",
    "ExtraTrees_PHYS": "Physics-guided extremely randomised trees",
    "LightGBM_PHYS": "Physics-guided LightGBM",
    "SVR_rbf_PHYS": "Physics-guided RBF support vector regression",
    "SVR_rbf": "RBF support vector regression",
    "RandomForest": "Random forest",
    "RandomForest_PHYS": "Physics-guided random forest",
    "NN_MLP(1000)": "Multilayer perceptron (1,000 epochs)",
    "KNN_PHYS": "Physics-guided k-nearest neighbours",
    "KNN": "k-nearest neighbours",
    "Ridge": "Ridge regression",
    "LinearRegression": "Linear regression",
    "Lasso": "Lasso regression",
    "ElasticNet": "Elastic net",
    "LinearRegression_PHYS": "Physics-guided linear regression",
    "Ridge_PHYS": "Physics-guided ridge regression",
    "Lasso_PHYS": "Physics-guided lasso regression",
    "ElasticNet_PHYS": "Physics-guided elastic net",
}

FIGURE_WIDTH = 16.0
FIGURE_HEIGHT_SCALE = np.sqrt(2.0)
STANDARD_COLOR = "#595959"
PHYSICS_COLOR = "#00796B"
AXIS_LABEL_SIZE = 36
TICK_LABEL_SIZE = 25
VALUE_LABEL_SIZE = 24


def _is_physics_guided(model):
    return model.endswith("_PHYS") or model.startswith("NN_PINN_HARD")


def main():
    if not os.path.exists(CVMETRICS):
        print("skipped: missing", CVMETRICS)
        return

    metrics = pd.read_csv(CVMETRICS).sort_values("R2").reset_index(drop=True)
    scores_pct = metrics["R2"] * 100.0
    base_height = 0.34 * len(metrics) + 1
    fig, axis = plt.subplots(
        figsize=(FIGURE_WIDTH, base_height * FIGURE_HEIGHT_SCALE)
    )
    fig._wag_layout_pad = 1.7

    physics_guided = metrics["model"].map(_is_physics_guided)
    colors = [
        PHYSICS_COLOR if guided else STANDARD_COLOR
        for guided in physics_guided
    ]
    labels = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])

    axis.barh(
        labels,
        scores_pct,
        height=0.72,
        color=colors,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
    )
    axis.axvline(0, color="black", linewidth=1.2)
    axis.axvline(100, color="none", linewidth=0)
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_xlabel("Five-fold cross-validated R² (%)", labelpad=14)
    axis._wag_axis_labelsize = AXIS_LABEL_SIZE
    axis._wag_tick_labelsize = TICK_LABEL_SIZE
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=10)
    axis.set_facecolor("#FBFBFB")

    for index, value in enumerate(scores_pct):
        label = axis.text(
            value - 1.2,
            index,
            f"{value:.1f}",
            ha="right",
            va="center",
            color="white",
        )
        label._wag_fontsize = VALUE_LABEL_SIZE

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

    save(fig, "Fig3_cv_r2_ranking")


if __name__ == "__main__":
    main()

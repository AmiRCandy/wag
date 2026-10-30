"""Figure 9: sensitivity of R-squared to the train/test split."""

import os

from common import METRICS, save

import matplotlib.pyplot as plt
import pandas as pd


def main():
    if not os.path.exists(METRICS):
        print("skipped: missing", METRICS)
        return
    metrics = pd.read_csv(METRICS)
    models = ["XGBoost", "ExtraTrees", "GradientBoosting", "NN_PINN_HARD", "LightGBM"]
    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for model in models:
        subset = metrics[metrics.model == model]
        if subset.empty:
            continue
        grouped = subset.groupby("split")["R2"].max()
        order = sorted(grouped.index, key=lambda value: int(value.split("/")[0]))
        axis.plot(order, [grouped[item] for item in order], "o-", label=model, lw=1.8, ms=6)
        plotted = True
    if not plotted:
        plt.close(fig)
        print("skipped: no top-model split metrics found")
        return
    axis.set_xlabel("train/test split")
    axis.set_ylabel("held-out R²")
    axis.legend(fontsize=9)
    save(fig, "Fig9_split_sensitivity")


if __name__ == "__main__":
    main()

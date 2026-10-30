"""Figure 7: residual analysis of the best model."""

from common import out_of_fold_predictions, prepare_model_data, save

import matplotlib.pyplot as plt
import seaborn as sns


HISTOGRAM_COLOR = "#0072B2"
KDE_COLOR = "#D62728"
SCATTER_COLOR = "#00796B"


def main():
    _, y, x, _, _, _ = prepare_model_data()
    predictions = out_of_fold_predictions(x, y)
    residuals = y - predictions

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig._wag_layout_pad = 0.8
    fig._wag_layout_w_pad = 0.7

    sns.histplot(
        residuals,
        bins=30,
        stat="density",
        kde=True,
        color=HISTOGRAM_COLOR,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.8,
        kde_kws={"bw_adjust": 1.0},
        line_kws={"linewidth": 2.6},
        ax=axes[0],
    )
    # Set the KDE explicitly so Seaborn cannot inherit the histogram color.
    if axes[0].lines:
        axes[0].lines[-1].set_color(KDE_COLOR)
        axes[0].lines[-1].set_linewidth(2.6)

    axes[0].axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    axes[0].set_xlabel("(a)\nIRF residual", labelpad=12)
    axes[0].set_ylabel("Probability density")

    axes[1].scatter(
        predictions,
        residuals,
        alpha=0.55,
        s=28,
        color=SCATTER_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[1].axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    axes[1].set_xlabel("(b)\nPredicted IRF (%)", labelpad=12)
    axes[1].set_ylabel("Residual")

    save(fig, "Fig7_residuals")


if __name__ == "__main__":
    main()

"""Figure 1: distributions of key simulation inputs and the target."""

from common import TARGET, load_data, save

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


PANELS = [
    ("effective_perm_md", "(a)\nEffective permeability (mD)", 0, False),
    ("init_pressure_psi", "(b)\nInitial reservoir pressure (psi)", 0, False),
    ("depth_ft", "(c)\nReservoir depth (ft)", 0, False),
    ("porosity_pct", "(d)\nPorosity (%)", 1, False),
    ("gas_rate_mscf", "(e)\nGas rate (Mscf/d)", 0, False),
    ("wat_rate_stb", "(f)\nWater rate (STB/d)", 0, False),
    ("wag_halfcycle_mo", "(g)\nWAG half-cycle (months)", 0, True),
    (TARGET, "(h)\nIncremental RF (%)", 1, False),
]

FIGURE_SCALE = np.sqrt(2.0)
AXIS_LABEL_SIZE = 36
TICK_LABEL_SIZE = 26
PANEL_COLOR = "#5A5A5A"


def _number_formatter(decimals):
    return FuncFormatter(
        lambda value, _position: f"{value:,.{decimals}f}"
    )


def _set_true_value_ticks(axis, values, decimals, discrete):
    """Show readable ticks spanning the exact range of the plotted data."""
    data_min = float(values.min())
    data_max = float(values.max())

    if discrete:
        ticks = np.sort(values.unique()).astype(int)
        axis.set_xlim(ticks[0] - 0.5, ticks[-1] + 0.5)
    else:
        ticks = np.linspace(data_min, data_max, 5)
        axis.set_xlim(data_min, data_max)

    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(_number_formatter(decimals))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.yaxis.set_major_formatter(_number_formatter(0))


def main():
    df, _, _, _ = load_data()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16 * FIGURE_SCALE, 6 * FIGURE_SCALE),
    )
    fig._wag_layout_pad = 1.5
    fig._wag_layout_h_pad = 3.0
    fig._wag_layout_w_pad = 2.0
    axes = np.atleast_1d(axes).ravel()

    for axis, (column, label, decimals, discrete) in zip(axes, PANELS):
        if column not in df.columns:
            axis.axis("off")
            continue

        values = df[column].dropna()
        if discrete:
            counts = values.value_counts().sort_index()
            axis.bar(
                counts.index,
                counts.values,
                width=0.82,
                color=PANEL_COLOR,
                edgecolor="white",
                linewidth=0.95,
                alpha=0.95,
            )
        else:
            axis.hist(
                values,
                bins=30,
                color=PANEL_COLOR,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.95,
                rwidth=0.96,
            )
        _set_true_value_ticks(axis, values, decimals, discrete)

        axis._wag_axis_labelsize = AXIS_LABEL_SIZE
        axis._wag_tick_labelsize = TICK_LABEL_SIZE
        axis.set_xlabel(label, labelpad=14)
        axis.xaxis.set_label_position("bottom")
        axis.tick_params(
            axis="x",
            labelbottom=True,
            labeltop=False,
            pad=8,
        )
        axis.tick_params(axis="y", pad=8)
        axis.set_ylabel("Count", labelpad=12)
        axis.set_facecolor("#FBFBFB")

    save(fig, "Fig1_data_distribution")


if __name__ == "__main__":
    main()

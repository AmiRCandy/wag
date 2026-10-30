"""Figure 8: precomputed held-out permutation importance of XGBoost."""

from common import repo_path, save

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


FEATURE_IMPORTANCE = repo_path(
    "WAG_FEATURE_IMPORTANCE",
    "results/feature_importance.csv",
)
BAR_COLOR = "#00796B"
ERROR_COLOR = "#404040"
TOP_FEATURES = 15

FEATURE_LABELS = {
    "perm_mult": "Permeability multiplier",
    "effective_perm_md": "Effective permeability (mD)",
    "porosity_frac": "Porosity (fraction)",
    "porosity_pct": "Porosity (%)",
    "init_pressure_psi": "Initial reservoir pressure (psi)",
    "depth_ft": "Reservoir depth (ft)",
    "gas_rate_mscf": "Gas injection rate (Mscf/d)",
    "wat_rate_stb": "Water injection rate (STB/d)",
    "prod_orat_stb": "Producer oil target (STB/d)",
    "wag_halfcycle_mo": "WAG half-cycle (months)",
    "wag_ratio_approx": "Approximate WAG ratio",
    "eng_gas_water_ratio": "Gas-to-water rate ratio",
    "eng_total_injection": "Total injection rate",
    "eng_perm_poro": "Permeability-porosity interaction",
    "eng_pressure_gradient": "Initial pressure-to-depth ratio",
    "eng_log_perm": "Log effective permeability",
    "eng_gas_per_perm": "Gas rate per permeability",
}


def _display_name(name):
    return FEATURE_LABELS.get(name, str(name).replace("_", " ").title())


def _load_importance():
    values = pd.read_csv(FEATURE_IMPORTANCE)
    required = {"feature", "importance_mean", "importance_std"}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(
            f"{FEATURE_IMPORTANCE} is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if values.empty:
        raise ValueError(f"{FEATURE_IMPORTANCE} contains no feature-importance rows")

    for column in ("importance_mean", "importance_std"):
        values[column] = pd.to_numeric(values[column], errors="raise")

    return (
        values.nlargest(TOP_FEATURES, "importance_mean")
        .sort_values("importance_mean")
        .reset_index(drop=True)
    )


def main():
    values = _load_importance()

    fig, axis = plt.subplots(figsize=(11.5, 8.5))
    fig._wag_layout_pad = 0.9
    axis._wag_axis_labelsize = 33
    axis._wag_tick_labelsize = 22

    positions = np.arange(len(values))
    axis.barh(
        positions,
        values["importance_mean"],
        xerr=values["importance_std"],
        height=0.72,
        color=BAR_COLOR,
        edgecolor="white",
        linewidth=0.9,
        error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.3, "capsize": 3},
    )
    axis.axvline(0, color="black", linewidth=1.2)
    axis.set_yticks(positions)
    axis.set_yticklabels([_display_name(name) for name in values["feature"]])
    axis.set_xlabel("Decrease in held-out R² after permutation")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))

    error_min = float(
        (values["importance_mean"] - values["importance_std"]).min()
    )
    error_max = float(
        (values["importance_mean"] + values["importance_std"]).max()
    )
    x_min = min(0.0, error_min)
    span = max(error_max - x_min, 1e-12)
    axis.set_xlim(x_min, error_max + 0.06 * span)
    axis.set_facecolor("#FBFBFB")

    save(fig, "Fig8_feature_importance")


if __name__ == "__main__":
    main()

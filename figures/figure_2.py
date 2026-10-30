"""Figure 2: Pearson correlation heatmap."""

from common import TARGET, load_data, save

import matplotlib.pyplot as plt
import seaborn as sns


DISPLAY_LABELS = {
    "perm_mult": "Perm. multiplier",
    "effective_perm_md": "Effective perm. (mD)",
    "porosity_frac": "Porosity (frac.)",
    "porosity_pct": "Porosity (%)",
    "init_pressure_psi": "Initial pressure (psi)",
    "depth_ft": "Reservoir depth (ft)",
    "gas_rate_mscf": "Gas inj. rate (Mscf/d)",
    "wat_rate_stb": "Water inj. rate (STB/d)",
    "prod_orat_stb": "Producer oil target (STB/d)",
    "wag_halfcycle_mo": "WAG half-cycle (mo)",
    "wag_ratio_approx": "Approx. WAG ratio (-)",
    "rf_incremental_pct": "Incremental RF (%)",
}


def main():
    df, _, numeric, _ = load_data()
    columns = [column for column in numeric if column in df.columns] + [TARGET]
    correlation = df[columns].corr(method="pearson").rename(
        index=DISPLAY_LABELS,
        columns=DISPLAY_LABELS,
    )
    fig, axis = plt.subplots(
        figsize=(1.5 * (1.0 + 0.6 * len(columns)), 1.5 * (0.8 + 0.6 * len(columns)))
    )
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="#D0D0D0",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
        ax=axis,
    )
    for text in axis.texts:
        text._wag_fontsize = 20
    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=90,
        ha="center",
        va="top",
    )
    axis.tick_params(axis="x", pad=10)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    save(fig, "Fig2_correlation_heatmap")


if __name__ == "__main__":
    main()

"""Shared data loading, modelling, and plot helpers for the figure scripts."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _ensure(package, pip_name=None):
    try:
        __import__(package)
    except Exception:
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or package],
            check=True,
        )


for _package, _pip_name in (
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
):
    _ensure(_package, _pip_name)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, NullLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(env_name, default):
    """Resolve defaults and relative environment overrides from the repository root."""
    path = Path(os.environ.get(env_name, default)).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


SIM_CSV = repo_path("WAG_SIM_CSV", "WAG_simulation_dataset_FULL.csv")
METRICS = repo_path("WAG_METRICS", "results/metrics.csv")
CVMETRICS = repo_path("WAG_CV", "results/cv_metrics.csv")
FIGDIR = repo_path("WAG_FIGDIR", "figures_output")
FONT_PATH = REPO_ROOT / "font_files" / "times.ttf"
TARGET = "rf_incremental_pct"
SEED = 42
PAL = "colorblind"

os.makedirs(FIGDIR, exist_ok=True)
if not FONT_PATH.is_file():
    raise FileNotFoundError(f"Bundled figure font not found: {FONT_PATH}")
font_manager.fontManager.addfont(str(FONT_PATH))
FIGURE_FONT = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "font.size": 22,
        "font.family": FIGURE_FONT,
        "axes.labelsize": 33,
        "axes.titlesize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)
sns.set_palette(PAL)


def _labels_are_numeric(labels):
    """Return True when an axis displays numerical rather than categorical labels."""
    visible = [label.get_text().strip().replace("−", "-") for label in labels]
    visible = [label for label in visible if label]
    if not visible:
        return False
    try:
        for label in visible:
            float(label.replace(",", ""))
    except ValueError:
        return False
    return True


def _style_axis(axis):
    """Apply the common publication style to one axes object."""
    axis.grid(False, which="both")
    axis.margins(x=0, y=0)
    axis.autoscale(enable=True, axis="both", tight=True)

    axis_label_size = getattr(axis, "_wag_axis_labelsize", 33)
    tick_label_size = getattr(axis, "_wag_tick_labelsize", 22)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    for text in axis.texts:
        text.set_fontproperties(
            font_manager.FontProperties(
                fname=str(FONT_PATH),
                size=getattr(text, "_wag_fontsize", 22),
            )
        )
    axis.xaxis.label.set_fontproperties(
        font_manager.FontProperties(fname=str(FONT_PATH), size=axis_label_size)
    )
    axis.yaxis.label.set_fontproperties(
        font_manager.FontProperties(fname=str(FONT_PATH), size=axis_label_size)
    )
    axis.title.set_fontproperties(
        font_manager.FontProperties(fname=str(FONT_PATH), size=22)
    )

    for dimension, labels in (
        (axis.xaxis, axis.get_xticklabels()),
        (axis.yaxis, axis.get_yticklabels()),
    ):
        numeric = _labels_are_numeric(labels)
        show_major_ticks = numeric or getattr(
            dimension,
            "_wag_show_ticks",
            False,
        )
        dimension.set_minor_locator(AutoMinorLocator() if numeric else NullLocator())
        dimension.set_tick_params(
            which="major",
            direction="in",
            length=8 if show_major_ticks else 0,
            width=1.5,
            labelsize=tick_label_size,
        )
        dimension.set_tick_params(
            which="minor",
            direction="in",
            length=4 if numeric else 0,
            width=1.2,
        )
        for label in labels:
            label.set_fontproperties(
                font_manager.FontProperties(fname=str(FONT_PATH), size=tick_label_size)
            )

    legend = axis.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(
                font_manager.FontProperties(fname=str(FONT_PATH), size=22)
            )


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width * 1.5, height * 1.5, forward=True)
    fig.canvas.draw()
    for axis in fig.axes:
        _style_axis(axis)
    fig.tight_layout(
        pad=getattr(fig, "_wag_layout_pad", 1.2),
        h_pad=getattr(fig, "_wag_layout_h_pad", None),
        w_pad=getattr(fig, "_wag_layout_w_pad", None),
    )
    fig.savefig(path + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".png")


def load_data():
    df = pd.read_csv(SIM_CSV)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = df[TARGET].astype(float).values

    excluded = {
        "source", "cum_oil_stb", "oip_initial_stb", "cum_gas_injected",
        # the target is rf_wag_total_pct - rf_waterflood_total_pct: using these as inputs leaks the answer
        "rf_wag_total_pct", "rf_waterflood_total_pct",
        # injected-volume columns written by STEP1 v5 (not inputs of the main model)
        "cum_solvent_injected_mscf", "cum_water_injected_stb", "hcpv_rb", "hcpvi_gas_frac",
        "pvi_water_frac", "wag_ratio_resvol", "hcpvi_source", "pressure_gradient_psi_ft",
        "hcpvi_target", "wag_ratio_target", "waterflood_years", "pvi_water_waterflood_frac",
        TARGET,
    }
    # same choice of injection inputs as STEP2 (WAG_INJ_FEATURES, default "volumes")
    if os.environ.get("WAG_INJ_FEATURES", "volumes") == "volumes" and {"hcpvi_gas_frac", "wag_ratio_resvol"} <= set(df.columns):
        excluded = (excluded - {"hcpvi_gas_frac", "wag_ratio_resvol"}) | {"gas_rate_mscf", "wat_rate_stb", "wag_ratio_approx"}
    features = [column for column in df.columns if column not in excluded]
    constants = [column for column in features if df[column].nunique(dropna=False) <= 1]
    variable = [column for column in features if column not in constants]
    numeric = [column for column in variable if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in variable if column not in numeric]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df, y, numeric, categorical


def make_preprocessor(numeric, categorical):
    transformers = [
        (
            "num",
            Pipeline(
                [("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]
            ),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def prepare_model_data():
    df, y, numeric, categorical = load_data()
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(df)
    if hasattr(x, "toarray"):
        x = x.toarray()
    return df, y, np.asarray(x, np.float32), numeric, categorical, preprocessor


def best_model():
    if HAS_XGB:
        return XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
    return ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)


def out_of_fold_predictions(x, y):
    return cross_val_predict(
        best_model(), x, y, cv=KFold(5, shuffle=True, random_state=SEED)
    )
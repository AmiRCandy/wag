"""Classical model registry and raw-input guided fitting; no work at import."""

from copy import deepcopy
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from ..config import PROJECT_ROOT
from ..features import CONTEXT_COLUMNS, FeatureSpec, make_model_pipeline, make_preprocessor
from ..runtime import dependency_preflight


FAMILIES = ("LinearRegression", "Ridge", "Lasso", "ElasticNet", "KNN", "SVR_rbf",
            "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM")
LOCKED_PARAMS = {"n_jobs", "nthread", "num_threads", "random_state", "device", "device_type",
                 "tree_method", "predictor", "gpu_id", "n_gpus"}


def load_classical_settings(path=PROJECT_ROOT / "configs" / "classical.json"):
    with Path(path).open(encoding="utf-8") as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict) or set(settings) != {"version", "enabled_models", "missing_model_policy", "variants", "guided", "models"} or settings["version"] != 1:
        raise ValueError("Invalid classical settings schema/version")
    names = settings["enabled_models"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names) or len(names) != len(set(names)) or any(name not in FAMILIES for name in names):
        raise ValueError("enabled_models must be unique known classical families")
    if settings["missing_model_policy"] not in ("error", "skip"):
        raise ValueError("missing_model_policy must be error or skip")
    if not isinstance(settings["variants"], list) or not settings["variants"] or any(not isinstance(v, str) for v in settings["variants"]) or len(set(settings["variants"])) != len(settings["variants"]) or any(v not in ("plain", "guided") for v in settings["variants"]):
        raise ValueError("variants must be unique plain/guided values")
    validate_guidance(settings["guided"])
    if not isinstance(settings["models"], dict) or set(settings["models"]) != set(FAMILIES):
        raise ValueError("Settings must declare all 11 families; enabled_models selects which run")
    for name, value in settings["models"].items():
        if not isinstance(value, dict) or set(value) != {"defaults", "search"} or not isinstance(value["defaults"], dict) or not isinstance(value["search"], dict):
            raise ValueError(f"Invalid defaults/search declaration for {name}")
        if (value["defaults"].keys() | value["search"].keys()) & LOCKED_PARAMS:
            raise ValueError("Thread/device/seed parameters are fixed by runtime policy, not configurable search dimensions")
        if any(not isinstance(v, list) or not v for v in value["search"].values()):
            raise ValueError("Search dimensions must be nonempty explicit lists")
    return settings


def validate_guidance(guidance):
    if not isinstance(guidance, dict) or set(guidance) != {"anchor_count", "anchor_weight", "gas_monotonicity", "bounds"}:
        raise ValueError("Invalid guided settings")
    count = guidance["anchor_count"]
    if type(count) is not int or count < 0 or count % 2:
        raise ValueError("anchor_count must be a nonnegative even integer (total rows)")
    weight = guidance["anchor_weight"]
    if type(weight) not in (int, float) or not np.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("anchor_weight must be in (0,1]")
    if type(guidance["gas_monotonicity"]) is not bool or (count and not guidance["gas_monotonicity"]):
        raise ValueError("Nonzero anchors require the explicit gas_monotonicity hypothesis")
    bounds = np.asarray(guidance["bounds"], dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not -100 <= bounds[0] < bounds[1] <= 100:
        raise ValueError("Incremental bounds must be an ordered interval within [-100,100]")


def assert_cpu_estimator(estimator):
    for name, value in estimator.get_params(deep=True).items():
        key = name.split("__")[-1]
        if key in ("n_jobs", "nthread", "num_threads") and value != 1:
            raise ValueError(f"{name} must be 1 for classical CPU fits")
        if key in ("device", "device_type") and value not in (None, "cpu"):
            raise ValueError("GPU trees are disabled")
        if key == "tree_method" and value not in (None, "hist"):
            raise ValueError("XGBoost tree_method must be hist on CPU")


def _estimator(name, defaults, seed):
    constructors = {"LinearRegression": LinearRegression, "Ridge": Ridge, "Lasso": Lasso,
                    "ElasticNet": ElasticNet, "KNN": KNeighborsRegressor, "SVR_rbf": SVR,
                    "RandomForest": RandomForestRegressor, "ExtraTrees": ExtraTreesRegressor,
                    "GradientBoosting": GradientBoostingRegressor}
    if name in constructors:
        estimator = constructors[name]()
        unknown = set(defaults) - set(estimator.get_params())
        if unknown:
            raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
        estimator.set_params(**defaults)
        policy = {key: value for key, value in (("n_jobs", 1), ("random_state", seed)) if key in estimator.get_params()}
        estimator.set_params(**policy)
    else:
        package, class_name = ("xgboost", "XGBRegressor") if name == "XGBoost" else ("lightgbm", "LGBMRegressor")
        versions = dependency_preflight([package])
        constructor = getattr(importlib.import_module(package), class_name)
        policy = {"n_jobs": 1, "random_state": seed}
        if name == "XGBoost":
            policy.update(tree_method="hist", verbosity=0)
            if int(versions[package].split(".")[0]) >= 2:
                policy["device"] = "cpu"
        else:
            policy.update(device_type="cpu", verbosity=-1, deterministic=True, force_col_wise=True)
        estimator = constructor(**defaults, **policy)
    assert_cpu_estimator(estimator)
    return estimator


@dataclass
class Registry:
    estimators: dict
    skipped: dict
    settings: dict


def build_registry(settings, seed=42):
    """Preflight configured families before fitting; explicit skip policy only."""
    estimators, skipped = {}, {}
    for name in settings["enabled_models"]:
        try:
            estimators[name] = _estimator(name, settings["models"][name]["defaults"], seed)
        except (ImportError, RuntimeError, OSError) as exc:
            if name not in ("XGBoost", "LightGBM") or settings["missing_model_policy"] == "error":
                raise
            skipped[name] = {"status": "skipped", "error_type": type(exc).__name__, "reason": str(exc)}
    return Registry(estimators, skipped, deepcopy(settings))


def supports_weights(estimator):
    return "sample_weight" in inspect.signature(estimator.fit).parameters


def training_targets(X, y):
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) != len(X) or not np.isfinite(values).all():
        raise ValueError("Training targets must be finite, one-dimensional and aligned with B rows")
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ValueError("Training target index is not aligned with B input index")
    return values


def raw_anchors(X, y, *, injection_mode, count, seed):
    """Resample real raw coordinates; change gas only; no permeability hypothesis."""
    spec = FeatureSpec(injection_mode)
    gas = {"volumes": "hcpvi_gas_frac", "design": "hcpvi_target", "rates": "gas_rate_mscf"}[injection_mode]
    observed = pd.to_numeric(X[gas], errors="raise").dropna()
    if not len(observed) or observed.min() == observed.max():
        raise ValueError("Gas-direction anchors require a nonconstant observed gas predictor")
    selected = spec.raw_features + CONTEXT_COLUMNS + ("source", "reservoir_family")
    rng = np.random.RandomState(seed)
    base = X.iloc[rng.randint(0, len(X), size=count // 2)].loc[:, list(selected)]
    low, high = base.copy(), base.copy()
    low[gas], high[gas] = observed.min(), observed.max()
    anchors = pd.concat([low, high], ignore_index=True)
    lower, upper = np.percentile(y, [10, 90])
    labels = np.concatenate([np.full(len(low), lower), np.full(len(high), upper)])
    return anchors, labels


class GuidedRegressor(RegressorMixin, BaseEstimator):
    """Raw-input weighted-anchor comparison with explicitly bounded inference.

    This is a gas-shape hypothesis, not a PDE PINN or a monotonicity guarantee.
    Preprocessing fits real training rows only; anchors never change its statistics.
    """

    def __init__(self, estimator, injection_mode="volumes", anchor_count=120,
                 anchor_weight=0.3, gas_monotonicity=True, bounds=(-100.0, 100.0), seed=42):
        self.estimator = estimator
        self.injection_mode = injection_mode
        self.anchor_count = anchor_count
        self.anchor_weight = anchor_weight
        self.gas_monotonicity = gas_monotonicity
        self.bounds = bounds
        self.seed = seed

    def fit(self, X, y):
        validate_guidance({"anchor_count": self.anchor_count, "anchor_weight": self.anchor_weight,
                           "gas_monotonicity": self.gas_monotonicity, "bounds": self.bounds})
        assert_cpu_estimator(self.estimator)
        if self.anchor_count and not supports_weights(self.estimator):
            raise ValueError("Weighted guided variant unsupported: estimator has no sample_weight; no unweighted fallback")
        y = training_targets(X, y)
        self.preprocessor_ = make_preprocessor(self.injection_mode)
        real = self.preprocessor_.fit_transform(X)
        self.estimator_ = clone(self.estimator)
        if self.anchor_count:
            anchors, labels = raw_anchors(X, y, injection_mode=self.injection_mode,
                                          count=self.anchor_count, seed=self.seed)
            synthetic = self.preprocessor_.transform(anchors)
            matrix = np.vstack([real, synthetic])
            targets = np.concatenate([y, labels])
            weights = np.concatenate([np.ones(len(y)), np.full(len(labels), self.anchor_weight)])
            self.estimator_.fit(matrix, targets, sample_weight=weights)
            self.anchor_target_quantiles_ = np.percentile(y, [10, 90]).tolist()
        else:
            self.estimator_.fit(real, y)
            self.anchor_target_quantiles_ = None
        self.fit_record_ = {"real_rows": len(y), "anchor_rows": self.anchor_count,
                            "anchor_weight": self.anchor_weight, "weighting_applied": bool(self.anchor_count),
                            "gas_monotonicity_hypothesis": bool(self.anchor_count), "permeability_monotonicity": False,
                            "bounds": list(self.bounds), "preprocessing_fit_rows": len(y), "seed": self.seed}
        return self

    def predict_raw(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self.preprocessor_.transform(X))

    def predict(self, X):
        return np.clip(self.predict_raw(X), self.bounds[0], self.bounds[1])


def make_classical_model(estimator, variant, injection_mode, guidance, seed):
    assert_cpu_estimator(estimator)
    if variant == "plain":
        return make_model_pipeline(estimator, injection_mode)
    if variant != "guided":
        raise ValueError("Unknown classical variant")
    return GuidedRegressor(clone(estimator), injection_mode=injection_mode, seed=seed, **guidance)

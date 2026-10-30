"""Explicit, fold-fitted sklearn preprocessing; import only after runtime bootstrap.

No file access, model fitting, worker creation or CUDA use occurs at import.
Every search must receive raw B DataFrames and the complete model pipeline.
"""

from dataclasses import asdict, dataclass
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .data import BASE_PREDICTORS, INJECTION_PREDICTORS


FEATURE_VERSION = "wag-features-1"
CONTEXT_COLUMNS = ("grid_nx", "grid_ny", "grid_nz", "cell_size_ft", "thickness_mult",
                   "kv_kh_mult", "mmp_full_psi", "tl_omega", "temperature_F",
                   "injectant", "process_type", "solvent_fraction", "primary_years", "wag_years")
UNITS = {"effective_perm_md": "mD", "porosity_frac": "fraction", "init_pressure_psi": "psi",
         "depth_ft": "ft", "prod_orat_stb": "STB/day", "wag_halfcycle_mo": "months",
         "waterflood_years": "years", "perm_contrast_exp": "dimensionless", "oil_visc_mult": "dimensionless",
         "hcpvi_gas_frac": "fraction of HCPV", "wag_ratio_resvol": "reservoir-volume water/gas ratio",
         "hcpvi_target": "fraction of HCPV", "wag_ratio_target": "reservoir-volume water/gas ratio",
         "gas_rate_mscf": "Mscf/day", "wat_rate_stb": "STB/day"}


class FeatureValidationError(ValueError):
    """Invalid predictors/context cannot be silently replaced with plausible values."""


@dataclass(frozen=True)
class FeatureSpec:
    injection_mode: str = "volumes"
    version: str = FEATURE_VERSION

    def __post_init__(self):
        if self.injection_mode not in INJECTION_PREDICTORS:
            raise ValueError("injection_mode must be volumes, design or rates")
        if self.version != FEATURE_VERSION:
            raise ValueError(f"Unsupported feature specification version: {self.version}")

    @property
    def raw_features(self):
        return BASE_PREDICTORS + INJECTION_PREDICTORS[self.injection_mode]

    @property
    def engineered_features(self):
        return ("log10_perm", "perm_times_porosity", "pressure_per_depth", "pressure_over_mmp") + tuple(
            "log1p_" + name for name in INJECTION_PREDICTORS[self.injection_mode])

    @property
    def all_features(self):
        return self.raw_features + self.engineered_features

    def description(self):
        injection = INJECTION_PREDICTORS[self.injection_mode]
        formulas = {"log10_perm": "log10(effective_perm_md / 1 mD)",
                    "perm_times_porosity": "effective_perm_md * porosity_frac",
                    "pressure_per_depth": "init_pressure_psi / depth_ft",
                    "pressure_over_mmp": "init_pressure_psi / saved fixed mmp_full_psi"}
        formulas.update({"log1p_" + name: f"ln(1 + {name} / reference_unit)" for name in injection})
        units = {name: UNITS[name] for name in self.raw_features}
        units.update(log10_perm="dimensionless", perm_times_porosity="mD", pressure_per_depth="psi/ft",
                     pressure_over_mmp="dimensionless")
        units.update({"log1p_" + name: "dimensionless" for name in injection})
        return {**asdict(self), "raw_features": list(self.raw_features),
                "engineered_features": list(self.engineered_features), "ordered_candidates": list(self.all_features),
                "units": units, "formulas": formulas,
                "injection_log_reference_units": {name: {"value": 1.0, "unit": UNITS[name]} for name in injection},
                "context_columns": list(CONTEXT_COLUMNS), "context_policy": "require saved fixed context",
                "missing_policy": "median-impute raw predictors inside training fold before feature engineering",
                "invalid_policy": "reject malformed/nonfinite values or domain violations; no clipping",
                "task": "conditional on realized injection" if self.injection_mode == "volumes" else "planned injection controls"}


def _frame(X):
    if not isinstance(X, pd.DataFrame):
        raise FeatureValidationError("Raw feature input must be a named pandas DataFrame")
    if X.columns.duplicated().any():
        raise FeatureValidationError("Duplicate input column names")
    if len(X) == 0:
        raise FeatureValidationError("Feature input has no records")
    return X


def _numeric(series, name, *, allow_missing):
    missing = series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())
    values = pd.to_numeric(series, errors="coerce").astype(float)
    malformed = (~missing & values.isna()) | np.isinf(values)
    booleans = series.map(lambda value: isinstance(value, (bool, np.bool_)))
    if malformed.any() or booleans.any() or (missing.any() and not allow_missing):
        raise FeatureValidationError(f"Invalid numeric values in {name}; missing values allowed={allow_missing}")
    return values


class RawFeatures(TransformerMixin, BaseEstimator):
    """Whitelist inputs and learn fixed context exclusively from the fit partition."""

    def __init__(self, injection_mode="volumes"):
        self.injection_mode = injection_mode

    def _inputs(self, X):
        X = _frame(X)
        spec = FeatureSpec(self.injection_mode)
        required = spec.raw_features + CONTEXT_COLUMNS
        missing = sorted(set(required) - set(X.columns))
        if missing:
            raise FeatureValidationError(f"Missing required predictor/context columns: {missing}")
        values = pd.DataFrame({name: _numeric(X[name], name, allow_missing=True)
                               for name in spec.raw_features}, index=X.index)
        for name in spec.raw_features:
            observed = values[name].dropna()
            if name in INJECTION_PREDICTORS[self.injection_mode]:
                valid = observed >= 0  # log1p(0) is defined; no invented epsilon.
            elif name == "porosity_frac":
                valid = (observed > 0) & (observed < 1)
            else:
                valid = observed > 0
            if not valid.all():
                raise FeatureValidationError(f"Invalid predictor domain in {name}; zero injection allowed, positive rock/design values required")
        context = {}
        for name in CONTEXT_COLUMNS:
            if name in ("injectant", "process_type"):
                series = X[name]
                if series.isna().any() or series.astype(str).str.strip().eq("").any():
                    raise FeatureValidationError(f"Missing fixed context in {name}")
                context[name] = series.astype(str)
            else:
                context[name] = _numeric(X[name], name, allow_missing=False)
        if (context["mmp_full_psi"] <= 0).any():
            raise FeatureValidationError("mmp_full_psi must be positive")
        return values, context

    def fit(self, X, y=None):
        X = _frame(X)
        # Metadata is a guard only. It never enters the predictor matrix.
        for name, value in (("source", "OPM_SPE5_simulation"), ("reservoir_family", "B")):
            if name not in X or not X[name].eq(value).all():
                raise FeatureValidationError("Preprocessing fit requires source=OPM_SPE5_simulation and reservoir_family=B")
        values, context = self._inputs(X)
        empty = [name for name in values if values[name].isna().all()]
        if empty:
            raise FeatureValidationError(f"No training-fold observations for predictors: {empty}; median imputation is undefined")
        learned = {}
        for name, series in context.items():
            if series.nunique(dropna=False) != 1:
                raise FeatureValidationError(f"Feature version {FEATURE_VERSION} requires fixed {name} inside the training fold")
            value = series.iloc[0]
            learned[name] = value.item() if hasattr(value, "item") else value
        self.context_ = learned
        self.n_fit_rows_ = len(X)
        self.feature_names_in_ = np.asarray(FeatureSpec(self.injection_mode).raw_features, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.observed_raw_ranges_ = {name: {"min": float(values[name].min()), "max": float(values[name].max()),
                                          "missing": int(values[name].isna().sum())} for name in values}
        return self

    def transform(self, X):
        check_is_fitted(self, "context_")
        values, context = self._inputs(X)
        changed = []
        for name, series in context.items():
            expected = self.context_[name]
            equal = series.eq(expected) if isinstance(expected, str) else np.isclose(series, expected, rtol=1e-10, atol=1e-10)
            if not np.all(equal):
                changed.append(name)
        if changed:
            raise FeatureValidationError(f"Fixed reservoir context differs from the saved training fold: {changed}; no context refit is permitted")
        # This extra column is a formula input only; DerivedFeatures never outputs it.
        values["mmp_full_psi"] = self.context_["mmp_full_psi"]
        return values

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "context_")
        return np.asarray(FeatureSpec(self.injection_mode).raw_features + ("mmp_full_psi",), dtype=object)


class DerivedFeatures(TransformerMixin, BaseEstimator):
    """Recompute all dependencies from raw, fold-imputed predictors."""

    def __init__(self, injection_mode="volumes"):
        self.injection_mode = injection_mode

    def fit(self, X, y=None):
        self.n_features_in_ = len(FeatureSpec(self.injection_mode).raw_features) + 1
        self.transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        spec = FeatureSpec(self.injection_mode)
        expected_names = spec.raw_features + ("mmp_full_psi",)
        if isinstance(X, pd.DataFrame) and tuple(X.columns) != expected_names:
            raise FeatureValidationError("Imputed raw feature order does not match the declared injection mode")
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_ or not np.isfinite(array).all():
            raise FeatureValidationError("Derived features require finite, correctly ordered imputed raw inputs")
        raw = {name: array[:, i] for i, name in enumerate(spec.raw_features)}
        for name, values in raw.items():
            valid = values >= 0 if name in INJECTION_PREDICTORS[self.injection_mode] else values > 0
            if name == "porosity_frac":
                valid &= values < 1
            if not valid.all():
                raise FeatureValidationError(f"Invalid imputed raw domain in {name}")
        if (array[:, -1] <= 0).any():
            raise FeatureValidationError("Invalid imputed MMP context")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                engineered = [np.log10(raw["effective_perm_md"]),
                              raw["effective_perm_md"] * raw["porosity_frac"],
                              raw["init_pressure_psi"] / raw["depth_ft"],
                              raw["init_pressure_psi"] / array[:, -1]]
                engineered.extend(np.log1p(raw[name]) for name in INJECTION_PREDICTORS[self.injection_mode])
                result = np.column_stack([array[:, :-1], *engineered])
        except FloatingPointError as exc:
            raise FeatureValidationError("Invalid/overflowing deterministic feature formula") from exc
        if not np.isfinite(result).all():
            raise FeatureValidationError("Feature engineering produced a nonfinite value")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(FeatureSpec(self.injection_mode).all_features, dtype=object)


class ConstantFeatures(TransformerMixin, BaseEstimator):
    """Fit-only constant selection. Inference cannot redefine retained columns."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
            raise FeatureValidationError("Constant selection requires a finite feature matrix")
        self.n_features_in_ = array.shape[1]
        self.active_mask_ = np.any(array != array[0], axis=0)
        self.constant_values_ = array[0, ~self.active_mask_].copy()
        if not self.active_mask_.any():
            raise FeatureValidationError("All candidate features are constant in this training fold")
        return self

    def transform(self, X):
        check_is_fitted(self, "active_mask_")
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_ or not np.isfinite(array).all():
            raise FeatureValidationError("Constant selector input does not match the fitted schema")
        return array[:, self.active_mask_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "active_mask_")
        names = np.asarray(input_features if input_features is not None else
                           [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        if len(names) != self.n_features_in_:
            raise FeatureValidationError("Feature names do not match constant selector schema")
        return names[self.active_mask_]


def make_preprocessor(injection_mode="volumes"):
    """Fresh, unfitted stages for one training partition; no parallel workers."""
    FeatureSpec(injection_mode)
    return Pipeline([("raw", RawFeatures(injection_mode)),
                     ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
                     ("engineer", DerivedFeatures(injection_mode)),
                     ("constants", ConstantFeatures()), ("scaler", StandardScaler())])


def make_model_pipeline(estimator, injection_mode="volumes"):
    """Use this complete raw-input pipeline INSIDE CV/search, never prefit X."""
    return Pipeline([("preprocess", make_preprocessor(injection_mode)), ("model", clone(estimator))])


def preprocessing_metadata(preprocessor):
    """JSON-safe inference specification; joblib retains the executable pipeline."""
    raw = preprocessor.named_steps["raw"]
    imputer = preprocessor.named_steps["imputer"]
    constants = preprocessor.named_steps["constants"]
    scaler = preprocessor.named_steps["scaler"]
    for stage in (raw, imputer, constants, scaler):
        check_is_fitted(stage)
    spec = FeatureSpec(raw.injection_mode)
    candidates = np.asarray(spec.all_features)
    active = candidates[constants.active_mask_].tolist()
    inactive = candidates[~constants.active_mask_].tolist()
    return {"specification": spec.description(), "fit_role": "B_training_partition",
            "fit_rows": raw.n_fit_rows_, "fixed_context": dict(raw.context_),
            "observed_raw_ranges": deepcopy(raw.observed_raw_ranges_),
            "median_imputation": dict(zip(raw.get_feature_names_out().tolist(), imputer.statistics_.tolist())),
            "active_features": active, "constant_features": dict(zip(inactive, constants.constant_values_.tolist())),
            "scaler": {"feature_order": active, "mean": scaler.mean_.tolist(),
                       "variance": scaler.var_.tolist(), "scale": scaler.scale_.tolist()},
            "stage_order": [name for name, _ in preprocessor.steps],
            "notes": ["Metadata is descriptive; load the trusted serialized pipeline for inference",
                      "Validation/test inputs do not change fitted statistics or feature order"]}

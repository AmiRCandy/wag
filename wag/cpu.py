"""Bounded classical search/fitting APIs; B experiment orchestration is Part 6."""

from copy import deepcopy
from dataclasses import dataclass, field
import traceback
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, RandomizedSearchCV
from threadpoolctl import threadpool_limits

from .models.classical import (LOCKED_PARAMS, assert_cpu_estimator, build_registry,
                               make_classical_model, supports_weights, training_targets)
from .runtime import cpu_parallelism, task_seed
from .progress import progress


@dataclass
class ModelResult:
    model_id: str
    status: str
    estimator: object = None
    details: dict = field(default_factory=dict)

    def record(self):
        return {"model_id": self.model_id, "status": self.status, **deepcopy(self.details)}


def _model_id(family, variant):
    return family if variant == "plain" else family + "_GUIDED"


def _failure(model_id, exc, seed):
    return ModelResult(model_id, "failed", details={"seed": seed,
        "error": {"type": type(exc).__name__, "message": str(exc),
                  "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=6))},
        "fallback_performed": False})


def _seeded(base, seed):
    from sklearn.base import clone
    estimator = clone(base)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    assert_cpu_estimator(estimator)
    return estimator


def _cv_indices(X, config, seed, cv):
    folds = list(KFold(config.inner_folds, shuffle=True, random_state=seed).split(X)) if cv is None else list(cv)
    if len(folds) < 2:
        raise ValueError("Inner tuning requires at least two folds")
    seen = []
    for train, valid in folds:
        for indices in (train, valid):
            array = np.asarray(indices)
            if array.ndim != 1 or array.dtype.kind not in "iu" or not len(array) or array.min() < 0 or array.max() >= len(X) or len(array) != len(set(array.tolist())):
                raise ValueError("Inner CV must use unique, valid positions within this B partition")
        if set(train) & set(valid) or set(train) | set(valid) != set(range(len(X))):
            raise ValueError("Inner training/validation overlap or fail to cover their B partition")
        seen.extend(valid)
    if len(seen) != len(set(seen)) or set(seen) != set(range(len(X))):
        raise ValueError("Each B row must appear in inner validation exactly once")
    return folds


def _prefix_space(space, estimator, variant):
    prefix = "model__" if variant == "plain" else "estimator__"
    valid = estimator.get_params(deep=True)
    result = {}
    for name, values in space.items():
        if name in LOCKED_PARAMS or "__" in name:
            raise ValueError("Search cannot change resource/device/seed policy or preprocessing")
        if prefix + name not in valid or not isinstance(values, list) or not values:
            raise ValueError(f"Invalid search dimension: {name}")
        result[prefix + name] = values
    return result


def _fit_one(family, variant, base, guidance, X, y, config, *, params=None, phase="fit", fold=0):
    model_id = _model_id(family, variant)
    seed = task_seed(config.seed, model_id, phase, fold)
    try:
        estimator = _seeded(base, seed)
        if params:
            if set(params) & LOCKED_PARAMS or any("__" in name for name in params):
                raise ValueError("Refit parameters cannot change runtime policy/preprocessing")
            estimator.set_params(**params)
        assert_cpu_estimator(estimator)
        model = make_classical_model(estimator, variant, config.injection_mode, guidance, seed)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
        details = {"seed": seed, "phase": phase, "fold": fold, "best_params": params or {},
                   "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
                   "native_threads": 1, "inner_cv_rmse": None, "search_performed": False,
                   "fallback_performed": False}
        if variant == "guided":
            details["guidance"] = model.fit_record_
        return ModelResult(model_id, "fitted", model, details)
    except Exception as exc:
        return _failure(model_id, exc, seed)


def search_models(X, y, config, settings, *, cv=None, search_spaces=None, fold=0):
    """Iterate families/variants sequentially; parallelize only candidate/fold fits.

    Returns in-memory fitted models plus explicit errors/skips/trial records.
    Inner RMSE is a tuning score, never an outer or external performance estimate.
    Optional search_spaces overrides are deliberate smoke/experiment settings.
    """
    registry = build_registry(settings, config.seed)  # dependency preflight before any fit
    if search_spaces and set(search_spaces) - set(settings["enabled_models"]):
        raise ValueError("Search-space overrides contain unknown/disabled families")
    y = training_targets(X, y)
    results = {}
    for family in settings["enabled_models"]:
        for variant in settings["variants"]:
            model_id = _model_id(family, variant)
            if family in registry.skipped:
                results[model_id] = ModelResult(model_id, "skipped", details=registry.skipped[family])
                continue
            base = registry.estimators[family]
            guidance = settings["guided"]
            if variant == "guided" and guidance["anchor_count"] and not supports_weights(base):
                results[model_id] = ModelResult(model_id, "skipped", details={"reason": "Weighted guided fitting unsupported: no sample_weight; no unweighted fallback"})
                continue
            seed = task_seed(config.seed, model_id, "search", fold)
            space = (search_spaces or {}).get(family, settings["models"][family]["search"])
            if not space:
                results[model_id] = _fit_one(family, variant, base, guidance, X, y, config, phase="default_fit", fold=fold)
                continue
            try:
                model = make_classical_model(_seeded(base, seed), variant, config.injection_mode, guidance, seed)
                grid = _prefix_space(space, model, variant)
                cv_seed = task_seed(config.seed, "split", "cpu_inner_cv", fold)
                indices = _cv_indices(X, config, cv_seed, cv)
                total = len(ParameterGrid(grid))
                common = dict(estimator=model, scoring="neg_root_mean_squared_error", cv=indices,
                              n_jobs=config.workers, pre_dispatch=config.workers, error_score="raise",
                              refit=True, return_train_score=False, verbose=1)
                if config.search == "grid":
                    search = GridSearchCV(param_grid=grid, **common)
                else:
                    search = RandomizedSearchCV(param_distributions=grid,
                                                n_iter=min(config.search_candidates, total), random_state=seed, **common)
                with cpu_parallelism(config.workers, config.threads), threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    search.fit(X, y)  # raw B DataFrame: every inner fold fits its own complete pipeline
                if not np.isfinite(search.cv_results_["mean_test_score"]).all():
                    raise ValueError("Nonfinite inner CV scores; no model selected")
                prefix = "model__" if variant == "plain" else "estimator__"
                unprefix = lambda p: {key[len(prefix):]: value for key, value in p.items()}
                trials = [{"candidate": i, "params": unprefix(params),
                           "mean_inner_rmse": -float(search.cv_results_["mean_test_score"][i]),
                           "std_inner_rmse": float(search.cv_results_["std_test_score"][i]),
                           "fold_inner_rmse": [-float(search.cv_results_[f"split{k}_test_score"][i]) for k in range(len(indices))]}
                          for i, params in enumerate(search.cv_results_["params"])]
                details = {"seed": seed, "phase": "search", "fold": fold,
                           "best_params": unprefix(search.best_params_), "inner_cv_rmse": -float(search.best_score_),
                           "score_role": "inner_tuning_only", "trials": trials, "search_performed": True,
                           "search": config.search, "requested_candidates": config.search_candidates,
                           "actual_candidates": len(trials), "inner_folds": len(indices), "cv_seed": cv_seed if cv is None else None,
                           "cv_source": "supplied_partition_indices" if cv is not None else "fixed_shuffled_KFold", "workers": config.workers,
                           "pre_dispatch": config.workers, "native_threads": 1, "fallback_performed": False,
                           "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
                           "warning_scope": "coordinator; worker warnings are also emitted by sklearn"}
                if variant == "guided":
                    details["guidance"] = search.best_estimator_.fit_record_
                results[model_id] = ModelResult(model_id, "fitted", search.best_estimator_, details)
            except Exception as exc:
                results[model_id] = _failure(model_id, exc, seed)
    return results


def fit_models(X, y, config, settings, *, selected_params=None, phase="refit", fold=0):
    """Fit independent declared CPU models in one bounded pool, without tuning."""
    registry = build_registry(settings, config.seed)
    expected_ids = {_model_id(family, variant) for family in settings["enabled_models"] for variant in settings["variants"]}
    if selected_params and set(selected_params) - expected_ids:
        raise ValueError("Selected parameters contain unknown/disabled model IDs")
    y = training_targets(X, y)
    tasks, results = [], {}
    for family in settings["enabled_models"]:
        for variant in settings["variants"]:
            model_id = _model_id(family, variant)
            if family in registry.skipped:
                results[model_id] = ModelResult(model_id, "skipped", details=registry.skipped[family])
            elif variant == "guided" and settings["guided"]["anchor_count"] and not supports_weights(registry.estimators[family]):
                results[model_id] = ModelResult(model_id, "skipped", details={"reason": "Weighted guided variant unsupported: no sample_weight"})
            else:
                tasks.append((family, variant, registry.estimators[family], (selected_params or {}).get(model_id, {})))
    with cpu_parallelism(config.workers, config.threads):
        stream = Parallel(n_jobs=config.workers, pre_dispatch=config.workers, return_as="generator")(
            delayed(_fit_one)(family, variant, base, settings["guided"], X, y, config,
                              params=params, phase=phase, fold=fold) for family, variant, base, params in tasks)
        with progress(stream, total=len(tasks), desc=f"CPU {phase}", unit="model") as bar:
            fitted = list(bar)
    results.update({result.model_id: result for result in fitted})
    return {_model_id(family, variant): results[_model_id(family, variant)]
            for family in settings["enabled_models"] for variant in settings["variants"]}

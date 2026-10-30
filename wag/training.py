"""B-only validation, selection and bounded task resume (no final/frozen models).

Outer folds run sequentially; classical search owns the only fitting pool.
Neural early stopping is inside development, followed by a fixed-epoch
development refit before outer scoring. A is never opened or fingerprinted.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import traceback
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from .progress import progress
from .config import PROJECT_ROOT
from .cpu import search_models
from .data import (TARGET, file_fingerprint, load_development, load_split_manifest,
                   make_split_manifest)
from .models.classical import build_registry, load_classical_settings
from .neural import fit_neural, inner_partition, load_neural_settings
from .runtime import configure_torch, dependency_preflight, git_identity, task_seed


SCHEMA = 1
MAX_TASK_ATTEMPTS = 2
PREDICTION_COLUMNS = ["row_id", "position", "csv_record_number", "model_id", "phase", "fold",
                      "observed", "prediction", "residual", "raw_prediction"]
METRIC_COLUMNS = ["model_id", "phase", "fold", "n", "R2", "RMSE", "MAE", "bias"]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Symlinked training output is not allowed")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_table(path, values, columns):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Symlinked training output is not allowed")
    pd.DataFrame(values, columns=columns).to_csv(temporary, index=False)
    os.replace(temporary, path)


@contextmanager
def _run_lock(directory):
    path = directory / ".training.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError("Run is locked. Verify no training process is active before manually removing a stale .training.lock") from exc
    try:
        yield
    finally:
        path.unlink()  # only our exclusively created transient lock


def regression_metrics(observed, prediction):
    observed, prediction = np.asarray(observed, dtype=float), np.asarray(prediction, dtype=float)
    if observed.ndim != 1 or prediction.shape != observed.shape or not len(observed) or not np.isfinite(observed).all() or not np.isfinite(prediction).all():
        raise ValueError("Scoring requires aligned, nonempty finite predictions/targets")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        error = prediction - observed
        r2 = None if len(observed) < 2 or np.ptp(observed) == 0 else float(r2_score(observed, prediction))
        values = {"n": len(observed), "R2": r2, "RMSE": float(np.sqrt(np.mean(error**2))),
                  "MAE": float(np.mean(np.abs(error))), "bias": float(error.mean())}
    if any(value is not None and not np.isfinite(value) for value in values.values()):
        raise ValueError("Scoring produced a nonfinite metric")
    return values


def _relative_folds(partition):
    lookup = {position: i for i, position in enumerate(partition["development"]["positions"])}
    return [(np.array([lookup[i] for i in inner["train"]["positions"]]),
             np.array([lookup[i] for i in inner["validation"]["positions"]]))
            for inner in partition["inner_folds"]]


def preflight_training(config):
    """All configured dependency/device checks before any task fits or run writes."""
    classical = neural = None
    versions = dependency_preflight(["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "tqdm"])
    if config.model_scope in ("all", "classical"):
        classical = load_classical_settings(config.classical_config)
        registry = build_registry(classical, config.seed)
        for family, package in (("XGBoost", "xgboost"), ("LightGBM", "lightgbm")):
            if family in classical["enabled_models"]:
                try:
                    versions[package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    versions[package] = None  # also bind explicitly skipped package state for resume
    device = {"requested": config.device, "selected": None, "reason": "classical-only experiment"}
    if config.model_scope in ("all", "neural"):
        neural = load_neural_settings(config.neural_config)
        versions.update(dependency_preflight(["torch"]))
        device = configure_torch(config.device)
    return classical, neural, versions, device


def _identity(config, development, classical, neural, versions):
    values = asdict(config)
    # A path is descriptive, never a development dependency. No A bytes are read.
    for key in ("run_dir", "resume", "test_csv"):
        values.pop(key)
    sources = {str(path.relative_to(PROJECT_ROOT)): file_fingerprint(path)
               for path in sorted((PROJECT_ROOT / "wag").rglob("*.py"))}
    sources["pyproject.toml"] = file_fingerprint(PROJECT_ROOT / "pyproject.toml")
    return {"schema": SCHEMA, "config": values, "B_fingerprint": development.fingerprint,
            "classical_settings": classical, "neural_settings": neural,
            "packages": versions, "source_fingerprints": sources,
            "selection_rule": "RMSE, then MAE, then model_id; nested uses mean outer-fold metrics",
            "task_attempt_limit": MAX_TASK_ATTEMPTS}


def _models(classical, neural):
    values = []
    if classical:
        for family in classical["enabled_models"]:
            for variant in classical["variants"]:
                values.append((family if variant == "plain" else family + "_GUIDED", "classical", family, variant))
    if neural:
        values += [(name, "neural", None, None) for name in neural["enabled_models"]]
    return values


def _failure(model_id, fold, exc):
    return {"model_id": model_id, "fold": fold, "status": "failed", "fallback_performed": False,
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=6))}}


def _execute_task(development, partition, config, classical, neural, model):
    model_id, kind, family, variant = model
    fold = partition["fold"]
    train_positions = partition["development"]["positions"]
    valid_positions = partition["validation"]["positions"]
    X = development.frame.iloc[train_positions]
    y = X[TARGET]
    base = {"model_id": model_id, "fold": fold, "phase": "B_outer_validation", "kind": kind,
            "training_row_ids": list(partition["development"]["row_ids"]),
            "validation_row_ids": list(partition["validation"]["row_ids"]), "fallback_performed": False}
    try:
        if kind == "classical":
            settings = deepcopy(classical)
            settings.update(enabled_models=[family], variants=[variant])
            result = search_models(X, y, config, settings, cv=_relative_folds(partition), fold=fold)[model_id]
            base["tuning"] = result.record()
            for trial in base["tuning"].get("trials", []):
                trial["trial_id"] = _digest([model_id, fold, "search", trial["candidate"], trial["params"]])
            base["selected_params"] = result.details.get("best_params", {})
        else:
            seed = task_seed(config.seed, "split", "neural_inner", fold)
            inner = inner_partition(len(X), neural, seed)
            result = fit_neural(X, y, config, neural, model_id, inner_split=inner, phase="inner", fold=fold)
            base["tuning"] = result.record()
            base["inner_training_row_ids"] = [development.row_ids[train_positions[i]] for i in inner[0]]
            base["inner_validation_row_ids"] = [development.row_ids[train_positions[i]] for i in inner[1]]
            if result.status == "fitted":
                base["selected_epochs"] = result.details["selected_epochs"]
                result = fit_neural(X, y, config, neural, model_id, epochs=base["selected_epochs"], phase="development_refit", fold=fold)
                base["development_refit"] = result.record()
        if result.status != "fitted":
            base.update(status=result.status, predictions=[], metrics=None)
            return base
        estimator = result.estimator
        heldout = development.frame.iloc[valid_positions]
        with threadpool_limits(limits=1):
            prediction = np.asarray(estimator.predict(heldout), dtype=float)
        scores = regression_metrics(heldout[TARGET], prediction)
        with threadpool_limits(limits=1):
            raw = np.asarray(estimator.predict_raw(heldout), dtype=float) if hasattr(estimator, "predict_raw") else None
        if raw is not None and (raw.shape != prediction.shape or not np.isfinite(raw).all()):
            raise ValueError("Raw guided predictions must be aligned and finite")
        rows = [{"row_id": development.row_ids[position], "position": position, "csv_record_number": position + 1,
                 "model_id": model_id, "phase": "B_outer_validation", "fold": fold,
                 "observed": float(development.frame[TARGET].iloc[position]), "prediction": float(prediction[i]),
                 "residual": float(development.frame[TARGET].iloc[position] - prediction[i]),
                 "raw_prediction": None if raw is None else float(raw[i])}
                for i, position in enumerate(valid_positions)]
        base.update(status="validated", predictions=rows, metrics=scores)
        # Capture held-out diagnostics while the actual development-fit model is
        # available. Plotting never substitutes the final model trained on all B.
        try:
            from .reporting import validation_importance
            base["importance"] = validation_importance(estimator, heldout, kind=kind, model_id=model_id,
                                                      seed=config.seed, fold=fold)
        except Exception as exc:
            base["importance"] = {"status": "failed", "error": _failure(model_id, fold, exc)["error"],
                                  "scope": "B_outer_validation", "fallback_performed": False}
        return base
    except Exception as exc:
        base.update(_failure(model_id, fold, exc), predictions=[], metrics=None)
        return base


def _validate_checkpoint(payload, development, partition, model_id, identity_hash, task_id):
    try:
        return _validate_checkpoint_contents(payload, development, partition, model_id, identity_hash, task_id)
    except (AttributeError, KeyError, TypeError, IndexError, FloatingPointError) as exc:
        raise ValueError(f"Malformed task checkpoint: {exc}") from exc


def _validate_checkpoint_contents(payload, development, partition, model_id, identity_hash, task_id):
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise ValueError("Task checkpoint requires the declared object schema")
    if payload.get("identity_hash") != identity_hash or payload.get("task_id") != task_id or payload.get("checksum") != _digest(payload.get("result")):
        raise ValueError("Task checkpoint identity/checksum mismatch; no task was reused")
    result = payload["result"]
    if result.get("model_id") != model_id or result.get("fold") != partition["fold"] or result.get("status") not in ("validated", "failed", "skipped") or result.get("task_id") != task_id or type(result.get("attempts")) is not int or not 1 <= result["attempts"] <= MAX_TASK_ATTEMPTS:
        raise ValueError("Task checkpoint model/fold/status mismatch")
    if result["status"] == "validated":
        rows = result["predictions"]
        positions = partition["validation"]["positions"]
        if len(rows) != len(positions):
            raise ValueError("Task checkpoint validation coverage mismatch")
        for row, position in zip(rows, positions):
            expected = {"position": position, "row_id": development.row_ids[position], "model_id": model_id,
                        "fold": partition["fold"], "observed": float(development.frame[TARGET].iloc[position]),
                        "csv_record_number": position + 1, "phase": "B_outer_validation"}
            if any(row.get(key) != value for key, value in expected.items()) or not np.isclose(row["residual"], row["observed"] - row["prediction"]):
                raise ValueError("Task checkpoint prediction row/target alignment mismatch")
        recomputed = regression_metrics([r["observed"] for r in rows], [r["prediction"] for r in rows])
        if recomputed != result["metrics"]:
            raise ValueError("Task checkpoint metrics do not match predictions")
    return result


def _task(directory, identity_hash, development, partition, config, classical, neural, model):
    task_id = _digest(["B_outer_validation", config.validation, partition["fold"], model[0]])
    path = directory / "tasks" / (task_id + ".json")
    attempt_path = directory / "tasks" / (task_id + ".attempt.json")
    cached = None
    if path.exists():
        if path.is_symlink():
            raise ValueError("Symlinked task checkpoint is not allowed")
        with path.open(encoding="utf-8") as stream:
            cached = _validate_checkpoint(json.load(stream), development, partition, model[0], identity_hash, task_id)
        if cached["status"] in ("validated", "skipped"):
            return cached
    attempts = 0
    if cached is not None and not attempt_path.exists():
        raise ValueError("Failed task has no attempt ledger; resume refused")
    if attempt_path.exists():
        if attempt_path.is_symlink():
            raise ValueError("Symlinked task attempt ledger is not allowed")
        with attempt_path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        if not isinstance(record, dict) or record.get("identity_hash") != identity_hash or record.get("task_id") != task_id or type(record.get("attempts")) is not int or not 1 <= record["attempts"] <= MAX_TASK_ATTEMPTS:
            raise ValueError("Task attempt ledger mismatch")
        attempts = record["attempts"]
        if cached is not None and attempts < cached["attempts"]:
            raise ValueError("Task attempt ledger predates its checkpoint")
    if attempts >= MAX_TASK_ATTEMPTS:
        if cached is not None:
            return cached
        result = {**_failure(model[0], partition["fold"], RuntimeError("Interrupted task exhausted the fixed two-attempt limit")),
                  "predictions": [], "metrics": None, "attempts": attempts, "task_id": task_id}
        _atomic_json(path, {"identity_hash": identity_hash, "task_id": task_id, "result": result, "checksum": _digest(result)})
        return result
    attempts += 1
    _atomic_json(attempt_path, {"identity_hash": identity_hash, "task_id": task_id, "attempts": attempts})
    result = _execute_task(development, partition, config, classical, neural, model)
    if cached is not None:
        result["prior_failed_attempts"] = cached.get("prior_failed_attempts", []) + [{"attempt": cached["attempts"],
            "error": cached.get("error") or cached.get("tuning", {}).get("error") or cached.get("development_refit", {}).get("error")}]
    result["task_id"], result["attempts"] = task_id, attempts
    _atomic_json(path, {"identity_hash": identity_hash, "task_id": task_id, "result": result, "checksum": _digest(result)})
    return result


def summarize_results(results, protocol, model_ids, fold_count, row_ids):
    fold_metrics, predictions, aggregate, ranking, excluded = [], [], [], [], []
    for name in model_ids:
        tasks = [r for r in results if r["model_id"] == name]
        complete = len(tasks) == fold_count and all(r["status"] == "validated" for r in tasks)
        rows = [row for task in tasks for row in task["predictions"]]
        for task in tasks:
            if task["status"] == "validated":
                fold_metrics.append({"model_id": name, "phase": "B_outer_validation", "fold": task["fold"], **task["metrics"]})
        predictions.extend(rows)
        if not complete:
            excluded.append({"model_id": name, "reason": "not validated in every required outer fold",
                             "fold_statuses": [{"fold": r["fold"], "status": r["status"]} for r in tasks]})
            continue
        if len({r["row_id"] for r in rows}) != len(rows) or (protocol == "nested-cv" and set(r["row_id"] for r in rows) != set(row_ids)):
            raise ValueError("OOF predictions must cover each B row exactly once per eligible model")
        pooled = regression_metrics([r["observed"] for r in rows], [r["prediction"] for r in rows])
        values = {"model_id": name, "phase": "B_validation_pooled", "fold": None, **pooled}
        aggregate.append(values)
        means = {}
        for metric in ("RMSE", "MAE", "R2", "bias"):
            scores = [r["metrics"][metric] for r in tasks]
            means["mean_outer_" + metric] = None if any(s is None for s in scores) else float(np.mean(scores))
            means["std_outer_" + metric] = None if any(s is None for s in scores) else float(np.std(scores))
        aggregate[-1].update(means)
        ranking.append({"model_id": name, "selection_RMSE": means["mean_outer_RMSE"] if protocol == "nested-cv" else pooled["RMSE"],
                        "selection_MAE": means["mean_outer_MAE"] if protocol == "nested-cv" else pooled["MAE"]})
    ranking.sort(key=lambda r: (r["selection_RMSE"], r["selection_MAE"], r["model_id"]))
    selection = {"schema_version": SCHEMA, "protocol": protocol,
                 "rule": "mean outer RMSE, then mean outer MAE, then model_id" if protocol == "nested-cv" else "B validation RMSE, then MAE, then model_id",
                 "primary_model_id": ranking[0]["model_id"] if ranking else None, "ranking": ranking,
                 "excluded_models": excluded, "winner_scope": "models validated in all required outer folds",
                 "A_outcomes_accessed": False, "models_frozen": False,
                 "interpretation": "B model-selection result, not an unbiased post-selection performance estimate",
                 "refit_plan": {"stage": "pending_Phase_7", "policy": "holdout: reuse development parameters/epochs; nested: tune on all B before final refit"}}
    return fold_metrics, predictions, aggregate, selection


def run_training(config, runtime=None):
    """Audit B, validate shared splits, checkpoint tasks and select using B only."""
    config = config.resolved()
    classical, neural, versions, device = preflight_training(config)
    development = load_development(config.train_csv, test_path=config.test_csv)
    identity = _identity(config, development, classical, neural, versions)
    identity_hash = _digest(identity)
    if config.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config = replace(config, run_dir=str(PROJECT_ROOT / "results" / "runs" / f"train_{stamp}_{uuid.uuid4().hex[:8]}"))
    directory = Path(config.run_dir)
    if config.resume and ((directory / "refit_manifest.json").exists() or (directory / "frozen_manifest.json").exists()):
        raise ValueError("Validation cannot resume after final refitting begins; use refit or train --resume")
    if config.resume:
        with (directory / "training_manifest.json").open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if not isinstance(previous, dict) or previous.get("schema_version") != SCHEMA or not isinstance(previous.get("artifact_fingerprints"), dict):
            raise ValueError("Resume requires a valid training manifest schema")
        if previous.get("models_frozen") or previous.get("final_refit_performed"):
            raise ValueError("Validation resume cannot modify final/frozen experiment state")
        if previous.get("identity_hash") != identity_hash or previous.get("identity") != identity:
            raise ValueError("Resume requires identical B data, configuration, settings, code and package versions")
        splits = load_split_manifest(directory / "split_manifest.json", development, config)
        if previous.get("split_fingerprint") != file_fingerprint(directory / "split_manifest.json"):
            raise ValueError("Saved split artifact changed; resume refused")
        with (directory / "data_audit.json").open(encoding="utf-8") as stream:
            if json.load(stream) != development.audit:
                raise ValueError("Saved B audit differs from current data; resume refused")
        for name, fingerprint in previous.get("artifact_fingerprints", {}).items():
            if Path(name).name != name or file_fingerprint(directory / name) != fingerprint:
                raise ValueError("Saved training artifact changed; resume refused")
    else:
        splits = make_split_manifest(development, config)
        directory.mkdir(parents=True, exist_ok=False)
    models = _models(classical, neural)
    manifest = {"schema_version": SCHEMA, "status": "running", "identity": identity, "identity_hash": identity_hash,
                "config": asdict(config), "packages": versions, "device": device, "git": git_identity(PROJECT_ROOT),
                "runtime": runtime, "B_rows": len(development.row_ids), "A_outcomes_accessed": False,
                "A_inputs_accessed": False, "models_frozen": False, "final_refit_performed": False,
                "importance_policy": {"scope": "B_outer_validation", "repeats": 5,
                                      "metric": "RMSE increase in percentage points and R2 decrease separately", "selection_use": False,
                                      "fit_performed": False, "CPU_threads": 1},
                "model_ids": [m[0] for m in models], "artifact_fingerprints": {}}
    with _run_lock(directory):
        if not config.resume:
            (directory / "tasks").mkdir()
            _atomic_json(directory / "split_manifest.json", splits)
            _atomic_json(directory / "data_audit.json", development.audit)
        if (directory / "tasks").is_symlink():
            raise ValueError("Symlinked task directory is not allowed")
        # Validate reusable checkpoints before changing an existing completion marker.
        if config.resume:
            for partition in splits["outer_partitions"]:
                for model in models:
                    task_id = _digest(["B_outer_validation", config.validation, partition["fold"], model[0]])
                    path = directory / "tasks" / (task_id + ".json")
                    if path.exists():
                        if path.is_symlink():
                            raise ValueError("Symlinked task checkpoint is not allowed")
                        with path.open(encoding="utf-8") as stream:
                            _validate_checkpoint(json.load(stream), development, partition, model[0], identity_hash, task_id)
        manifest["split_fingerprint"] = file_fingerprint(directory / "split_manifest.json")
        _atomic_json(directory / "training_manifest.json", manifest)
        results = []
        tasks = [(partition, model) for partition in splits["outer_partitions"] for model in models]
        with progress(tasks, desc="B validation", unit="model task") as bar:
            for partition, model in bar:
                bar.set_postfix(fold=partition["fold"], model=model[0], refresh=True)
                print(f"[train] outer fold {partition['fold']} | {model[0]}", flush=True)
                results.append(_task(directory, identity_hash, development, partition, config, classical, neural, model))
        folds, predictions, aggregates, selection = summarize_results(results, config.validation, [m[0] for m in models],
                                                                     len(splits["outer_partitions"]), development.row_ids)
        metric_extras = [f"{stat}_outer_{metric}" for metric in ("RMSE", "MAE", "R2", "bias") for stat in ("mean", "std")]
        prediction_name = "cv_oof_predictions.csv" if config.validation == "nested-cv" else "validation_predictions.csv"
        tables = [("validation_metrics.csv", aggregates, METRIC_COLUMNS + metric_extras),
                  (prediction_name, predictions, PREDICTION_COLUMNS)]
        if config.validation == "nested-cv":
            tables.append(("cv_fold_metrics.csv", folds, METRIC_COLUMNS))
        for name, values, columns in tables:
            _atomic_table(directory / name, values, columns)
        selection["task_selection_records"] = [{"model_id": r["model_id"], "fold": r["fold"], "status": r["status"],
                                                "selected_params": r.get("selected_params"), "selected_epochs": r.get("selected_epochs"),
                                                "task_id": r.get("task_id")} for r in results]
        _atomic_json(directory / "selection.json", selection)
        _atomic_json(directory / "task_records.json", results)
        history = []
        for task in results:
            for stage in ("tuning", "development_refit"):
                record = task.get(stage, {})
                for row in record.get("history", []):
                    history.append({"model_id": task["model_id"], "fold": task["fold"], "stage": stage, **row})
        _atomic_table(directory / "training_history.csv", history,
                      ["model_id", "fold", "stage", "epoch", "loss_pre_update", "data_mse_pre_update", "real_shape_penalty",
                       "collocation_shape_penalty", "train_mse_post_update", "inner_validation_rmse"])
        artifact_names = [name for name, _, _ in tables] + ["selection.json", "task_records.json", "training_history.csv", "data_audit.json", "split_manifest.json"]
        manifest.update(status="validation_complete" if all(r["status"] != "failed" for r in results) and selection["primary_model_id"] else "validation_incomplete",
                        primary_model_id=selection["primary_model_id"], failures=sum(r["status"] == "failed" for r in results),
                        skipped=sum(r["status"] == "skipped" for r in results),
                        artifact_fingerprints={name: file_fingerprint(directory / name) for name in artifact_names})
        _atomic_json(directory / "training_manifest.json", manifest)  # completion marker last; never a frozen-model marker
    return {"status": manifest["status"], "run_dir": str(directory), "B_rows": len(development.row_ids),
            "primary_model_id": selection["primary_model_id"], "failures": manifest["failures"], "skipped": manifest["skipped"],
            "A_outcomes_accessed": False, "final_refit_performed": False, "models_frozen": False}


def _verified_validation(config, development, current_identity):
    """Bind completed B results to unchanged fitting code, data and scientific settings."""
    directory = Path(config.run_dir)
    with (directory / "training_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("status") != "validation_complete" or manifest.get("identity_hash") != _digest(manifest.get("identity")):
        raise ValueError("Final refitting requires a complete, consistent B validation manifest")
    original = manifest["identity"]
    if {k: v for k, v in original.items() if k != "source_fingerprints"} != {k: v for k, v in current_identity.items() if k != "source_fingerprints"}:
        raise ValueError("B data/configuration/settings/packages differ from validation")
    # New orchestration/serialization code is recorded separately. Fit implementations
    # must remain byte-identical, permitting a completed Phase 6 run to be finalized.
    for path, fingerprint in original["source_fingerprints"].items():
        if path not in ("wag/training.py", "wag/cli.py") and current_identity["source_fingerprints"].get(path) != fingerprint:
            raise ValueError("Fitting/preprocessing code changed since validation; repeat B development")
    for name, fingerprint in manifest["artifact_fingerprints"].items():
        if Path(name).name != name or file_fingerprint(directory / name) != fingerprint:
            raise ValueError("Validation artifact fingerprint mismatch")
    splits = load_split_manifest(directory / "split_manifest.json", development, config)
    with (directory / "task_records.json").open(encoding="utf-8") as stream:
        results = json.load(stream)
    with (directory / "selection.json").open(encoding="utf-8") as stream:
        selection = json.load(stream)
    models = manifest["model_ids"]
    if len(results) != len(models) * len(splits["outer_partitions"]):
        raise ValueError("Incomplete validation task inventory")
    for partition in splits["outer_partitions"]:
        for name in models:
            candidates = [r for r in results if r["fold"] == partition["fold"] and r["model_id"] == name]
            if len(candidates) != 1:
                raise ValueError("Duplicate/missing validation model/fold records")
            result = candidates[0]
            task_id = _digest(["B_outer_validation", config.validation, partition["fold"], name])
            with (directory / "tasks" / (task_id + ".json")).open(encoding="utf-8") as stream:
                cached = _validate_checkpoint(json.load(stream), development, partition, name, manifest["identity_hash"], task_id)
            if cached != result:
                raise ValueError("Validation records differ from completed task checkpoints")
    recomputed = summarize_results(results, config.validation, models, len(splits["outer_partitions"]), development.row_ids)[-1]
    if any(selection.get(key) != recomputed[key] for key in ("primary_model_id", "ranking", "excluded_models", "rule")):
        raise ValueError("B-selected winner/ranking cannot be reproduced")
    if manifest["primary_model_id"] != selection["primary_model_id"] or not selection["primary_model_id"]:
        raise ValueError("Validation winner metadata mismatch")
    return manifest, splits, selection, results


def _refit_configuration(model, classical, neural, params=None, epochs=None):
    name, kind, family, variant = model
    if kind == "classical":
        return {"family": family, "variant": variant, "defaults": deepcopy(classical["models"][family]["defaults"]),
                "selected_params": deepcopy(params or {}), "guided": deepcopy(classical["guided"]) if variant == "guided" else None}
    return {"settings": deepcopy(neural), "selected_epochs": epochs}


def _selected_plan(model, config, development, splits, validation_results, classical, neural):
    name, kind, family, variant = model
    if config.validation == "holdout":
        selected = next(r for r in validation_results if r["model_id"] == name and r["fold"] == 0)
        if kind == "classical":
            return {"status": "selected", "source": "holdout_development_selection",
                    "selected_params": selected["selected_params"], "selection_task_id": selected["task_id"]}
        epochs = selected["selected_epochs"]
        if type(epochs) is not int or not 1 <= epochs <= neural["max_epochs"] or selected["tuning"].get("selected_epochs") != epochs or selected["development_refit"].get("selected_epochs") != epochs:
            raise ValueError("Neural epoch provenance does not match B inner selection")
        return {"status": "selected", "source": "holdout_B_inner_epoch_selection", "selected_epochs": epochs,
                "selection_task_id": selected["task_id"]}
    X, y = development.frame, development.frame[TARGET]
    if kind == "classical":
        settings = deepcopy(classical)
        settings.update(enabled_models=[family], variants=[variant])
        cv = [(np.asarray(f["train"]["positions"]), np.asarray(f["validation"]["positions"])) for f in splits["all_B_inner_folds"]]
        # Dedicated final-tuning seed namespace, not an outer-fold winner shortcut.
        tuning_config = replace(config, seed=task_seed(config.seed, name, "all_B_inner_tuning"))
        result = search_models(X, y, tuning_config, settings, cv=cv, fold=0)[name]
        return {"status": "selected" if result.status == "fitted" else "failed", "source": "all_B_inner_tuning",
                "selected_params": result.details.get("best_params"), "tuning": result.record()}
    result = fit_neural(X, y, config, neural, name, phase="all_B_inner_epoch_selection", fold=0)
    return {"status": "selected" if result.status == "fitted" else "failed", "source": "all_B_inner_epoch_selection",
            "selected_epochs": result.details.get("selected_epochs"), "tuning": result.record()}


def final_refit(config):
    """Fit each eligible comparison on all B and atomically freeze verified bundles.

    Requires completed B validation; never reads A. Partial final fits are not a
    frozen experiment. Explicit resume reuses saved models without retuning.
    """
    from .artifacts import (FROZEN_NAME, OOD_POLICY, atomic_json, load_frozen_models,
                            load_model_bundle, prediction_probe, publish_frozen_manifest, save_model_bundle)
    from .cpu import _fit_one
    from .runtime import cpu_parallelism
    from joblib import Parallel, delayed
    config = config.resolved()
    if config.run_dir is None:
        raise ValueError("Final refitting requires an explicit completed validation run_dir")
    directory = Path(config.run_dir)
    classical, neural, versions, device = preflight_training(config)
    development = load_development(config.train_csv, test_path=config.test_csv)
    identity = _identity(config, development, classical, neural, versions)
    validation, splits, selection, validation_results = _verified_validation(config, development, identity)
    eligible = {r["model_id"] for r in selection["ranking"]}
    models = [model for model in _models(classical, neural) if model[0] in eligible]
    if {m[0] for m in models} != eligible:
        raise ValueError("Eligible models do not match the configured comparison")
    identity["validation_manifest_fingerprint"] = file_fingerprint(directory / "training_manifest.json")
    identity["selection_fingerprint"] = file_fingerprint(directory / "selection.json")
    identity_hash = _digest(identity)
    if (directory / FROZEN_NAME).exists():
        frozen = load_frozen_models(directory, device="cpu")
        if frozen.manifest["refit_identity_hash"] != identity_hash:
            raise ValueError("Frozen run differs from requested final refit; no overwrite permitted")
        return {"status": "frozen", "run_dir": str(directory), "B_rows": len(development.row_ids),
                "primary_model_id": frozen.manifest["primary_model_id"], "models_frozen": True,
                "final_refit_performed": True, "A_outcomes_accessed": False}
    state_path = directory / "refit_manifest.json"
    if state_path.exists():
        if not config.resume:
            raise ValueError("Partial refit exists; use explicit --resume with identical settings")
        with state_path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("identity_hash") != identity_hash or previous.get("identity") != identity:
            raise ValueError("Final refit resume requires identical data/settings/code/packages/selection")
    probe = prediction_probe(development, config.injection_mode)
    records, entries, plans, pending = {}, {}, {}, []
    with _run_lock(directory):
        (directory / "final_models").mkdir(exist_ok=True)
        (directory / "refit_tasks").mkdir(exist_ok=True)
        if (directory / "final_models").is_symlink() or (directory / "refit_tasks").is_symlink():
            raise ValueError("Symlinked final artifact directories are not allowed")
        atomic_json(directory / "refit_probe.json", probe)
        state = {"schema_version": 1, "status": "refitting", "identity": identity, "identity_hash": identity_hash,
                 "B_rows": len(development.row_ids), "model_ids": [m[0] for m in models],
                 "primary_model_id": selection["primary_model_id"], "models_frozen": False,
                 "A_outcomes_accessed": False, "device": device}
        atomic_json(state_path, state)
        for model in models:
            name = model[0]
            token = _digest(["final_all_B", name])
            path = directory / "refit_tasks" / (token + ".json")
            cached = None
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    payload = json.load(stream)
                if payload.get("identity_hash") != identity_hash or payload.get("checksum") != _digest(payload.get("record")):
                    raise ValueError("Final task checkpoint mismatch")
                cached = payload["record"]
                if (not isinstance(cached, dict) or cached.get("model_id") != name
                        or cached.get("kind") != model[1] or cached.get("status") not in ("saved", "failed")
                        or type(cached.get("attempts")) is not int
                        or not 1 <= cached["attempts"] <= MAX_TASK_ATTEMPTS):
                    raise ValueError("Final task checkpoint schema mismatch")
            ledger = directory / "refit_tasks" / (token + ".attempt.json")
            attempts = 0
            if ledger.exists():
                with ledger.open(encoding="utf-8") as stream:
                    attempt = json.load(stream)
                if attempt.get("identity_hash") != identity_hash or type(attempt.get("attempts")) is not int or not 1 <= attempt["attempts"] <= MAX_TASK_ATTEMPTS:
                    raise ValueError("Final task attempt ledger mismatch")
                attempts = attempt["attempts"]
            if cached is not None and (not attempts or attempts < cached["attempts"]):
                raise ValueError("Final attempt ledger is missing or predates its checkpoint")
            if cached is not None and cached["status"] == "saved":
                if cached["entry"].get("model_id") != name or cached["entry"].get("kind") != model[1]:
                    raise ValueError("Saved final model identity mismatch")
                load_model_bundle(directory, cached["entry"], device="cpu")
                records[name], entries[name], plans[name] = cached, cached["entry"], cached["plan"]
                continue
            if attempts >= MAX_TASK_ATTEMPTS:
                records[name] = cached or {"model_id": name, "status": "failed", "error": "Interrupted final task exhausted two attempts", "attempts": attempts}
                continue
            attempts += 1
            atomic_json(ledger, {"identity_hash": identity_hash, "attempts": attempts})
            # A successful plan survives a fitting interruption and is never retuned.
            plan_path = directory / "refit_tasks" / (token + ".plan.json")
            if plan_path.exists():
                with plan_path.open(encoding="utf-8") as stream:
                    saved = json.load(stream)
                if saved.get("identity_hash") != identity_hash or saved.get("checksum") != _digest(saved.get("plan")):
                    raise ValueError("Final tuning-plan checkpoint mismatch")
                plan = saved["plan"]
            else:
                plan = _selected_plan(model, config, development, splits, validation_results, classical, neural)
                if plan["status"] == "selected":
                    atomic_json(plan_path, {"identity_hash": identity_hash, "plan": plan, "checksum": _digest(plan)})
            plans[name] = plan
            record = {"model_id": name, "kind": model[1], "status": "pending", "attempts": attempts, "plan": plan,
                      "fallback_performed": False}
            if plan["status"] != "selected":
                record["status"] = "failed"
                records[name] = record
                atomic_json(path, {"identity_hash": identity_hash, "record": record, "checksum": _digest(record)})
            else:
                pending.append((model, record, path))
        CPU = [(model, record, path) for model, record, path in pending if model[1] == "classical"]
        fitted = {}
        if CPU:
            registry = build_registry(classical, config.seed)
            with cpu_parallelism(config.workers, config.threads):
                stream = Parallel(n_jobs=config.workers, pre_dispatch=config.workers, return_as="generator")(
                    delayed(_fit_one)(model[2], model[3], registry.estimators[model[2]], classical["guided"],
                                      development.frame, development.frame[TARGET], config,
                                      params=record["plan"]["selected_params"], phase="final_all_B", fold=0)
                    for model, record, path in CPU)
                with progress(stream, total=len(CPU), desc="All-B CPU fits", unit="model") as bar:
                    results = list(bar)
            fitted.update({result.model_id: result for result in results})
        with progress(pending, desc="All-B save/refit", unit="model") as bar:
            for model, record, path in bar:
                name, kind = model[:2]
                bar.set_postfix(model=name, refresh=True)
                print(f"[refit] {name} | all {len(development.row_ids)} B rows", flush=True)
                result = fitted[name] if kind == "classical" else fit_neural(development.frame, development.frame[TARGET], config, neural,
                                  name, epochs=record["plan"]["selected_epochs"], phase="final_all_B", fold=0)
                record["fit"] = result.record()
                record["status"] = "failed"
                if result.status == "fitted":
                    try:
                        configuration = _refit_configuration(model, classical, neural, record["plan"].get("selected_params"), record["plan"].get("selected_epochs"))
                        entry = save_model_bundle(directory, name, kind, result.estimator, configuration, result.record(), probe)
                        if fitted_preprocessor_rows(result.estimator, kind) != len(development.row_ids):
                            raise ValueError("Final preprocessing does not cover all B")
                        record.update(status="saved", entry=entry)
                        entries[name] = entry
                    except Exception as exc:
                        record["serialization_error"] = _failure(name, 0, exc)["error"]
                records[name] = record
                atomic_json(path, {"identity_hash": identity_hash, "record": record, "checksum": _digest(record)})
        atomic_json(directory / "refit_plan.json", {"protocol": config.validation, "plans": plans,
                    "primary_model_id": selection["primary_model_id"], "selection_fingerprint": identity["selection_fingerprint"]})
        atomic_json(directory / "refit_records.json", records)
        complete = len(entries) == len(models) and all(r["status"] == "saved" for r in records.values())
        state.update(status="refit_complete" if complete else "refit_incomplete", final_refit_performed=complete,
                     model_config_hashes={name: entry["configuration_hash"] for name, entry in entries.items()})
        atomic_json(state_path, state)
        if complete:
            fingerprints = {path: fingerprint for entry in entries.values() for path, fingerprint in entry["fingerprints"].items()}
            fingerprints.update({name: file_fingerprint(directory / name) for name in
                ("training_manifest.json", "selection.json", "task_records.json", "split_manifest.json", "data_audit.json",
                 "refit_manifest.json", "refit_plan.json", "refit_records.json", "refit_probe.json")})
            frozen = {"schema_version": 1, "status": "frozen", "models_frozen": True,
                      "B": {"rows": len(development.row_ids), "fingerprint": development.fingerprint,
                            "row_ids_hash": _digest(list(development.row_ids)), "source": "OPM_SPE5_simulation", "family": "B"},
                      "primary_model_id": selection["primary_model_id"], "model_ids": [m[0] for m in models], "models": entries,
                      "refit_identity_hash": identity_hash, "validation_identity_hash": validation["identity_hash"],
                      "protocol": config.validation, "packages": versions, "config": asdict(config),
                      "skipped_comparisons": selection["excluded_models"], "OOD_policy": deepcopy(OOD_POLICY),
                      "A_outcomes_accessed": False, "A_inputs_accessed": False, "probe_file": "refit_probe.json",
                      "artifact_fingerprints": fingerprints}
            publish_frozen_manifest(directory, frozen)
    return {"status": "frozen" if complete else "refit_incomplete", "run_dir": str(directory),
            "B_rows": len(development.row_ids), "primary_model_id": selection["primary_model_id"],
            "models_frozen": complete, "final_refit_performed": complete, "A_outcomes_accessed": False}


def fitted_preprocessor_rows(estimator, kind):
    from .artifacts import fitted_preprocessor
    return fitted_preprocessor(estimator, kind).named_steps["raw"].n_fit_rows_

"""External simulated A evaluation of frozen models; no development or fitting.

Import performs no file access, worker creation or CUDA initialization. Each
explicit evaluation creates a new event; training/frozen artifacts stay intact.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from .artifacts import (FROZEN_NAME, OOD_POLICY, atomic_json, fitted_preprocessor,
                        load_frozen_models)
from .data import (DESIGN_COLUMNS, NUMERIC_COLUMNS, REQUIRED_COLUMNS, TARGET,
                   TEXT_COLUMNS, DataIntegrityError,
                   _check_physical_values, _check_text, _numeric, _read_columns,
                   _require_columns, _signatures, csv_header, file_fingerprint)
from .features import CONTEXT_COLUMNS, FeatureSpec


PHASE = "A_external_simulation_test"
PREDICTION_COLUMNS = ["row_id", "position", "csv_record_number", "model_id", "phase",
                      "is_B_selected_primary", "observed", "prediction", "residual",
                      "raw_prediction", "raw_residual", "ood_raw_range", "ood_scaled",
                      "ood_any", "raw_range_features", "scaled_features"]
METRIC_COLUMNS = ["model_id", "phase", "is_B_selected_primary", "output", "subgroup",
                  "n", "R2", "RMSE", "MAE", "bias"]


def _assert_A(frame):
    if not len(frame):
        raise DataIntegrityError("External A CSV contains no records")
    for column, expected in (("source", "OPM_SPE5_simulation"), ("reservoir_family", "A")):
        if not frame[column].eq(expected).all():
            raise DataIntegrityError(f"External evaluation requires {column}={expected!r} in every row")


def load_external(path, frozen):
    """Strict A integrity check after freezing; reject the entire invalid CSV.

    IDs exclude outcomes. No row filtering, target clipping or imputation here.
    Both recovery totals are integrity checks only, never predictor columns.
    """
    path = Path(path).resolve()
    if str(path) == frozen.manifest["B"]["fingerprint"]["path"]:
        raise DataIntegrityError("The B development path cannot be evaluated as A")
    before = file_fingerprint(path)
    header = csv_header(path)
    _require_columns(header, REQUIRED_COLUMNS)
    _assert_A(_read_columns(path, ("source", "reservoir_family")))
    raw = _read_columns(path, REQUIRED_COLUMNS)
    _assert_A(raw)
    _check_text(raw, TEXT_COLUMNS)
    frame = _numeric(raw, NUMERIC_COLUMNS)
    _check_physical_values(frame)
    identities = {
        "incremental_equals_wag_minus_waterflood": (frame[TARGET], frame.rf_wag_total_pct - frame.rf_waterflood_total_pct),
        "porosity_pct_equals_fraction_times_100": (frame.porosity_pct, frame.porosity_frac * 100),
        "effective_perm_equals_perm_mult_times_250": (frame.effective_perm_md, frame.perm_mult * 250)}
    errors = {}
    for name, (actual, expected) in identities.items():
        valid = np.isclose(actual, expected, rtol=1e-8, atol=1e-7)
        errors[name] = float(np.max(np.abs(actual - expected)))
        if not valid.all():
            raise DataIntegrityError(f"Arithmetic identity {name} failed at positions {np.flatnonzero(~valid).tolist()}")
    signatures = _signatures(frame, DESIGN_COLUMNS)
    if len(set(signatures)) != len(signatures):
        raise DataIntegrityError("Duplicate A design coordinates; no rows were removed")
    after = file_fingerprint(path)
    if before != after:
        raise DataIntegrityError("A CSV changed during loading")
    audit = {"schema_version": 1, "role": PHASE, "rows": len(frame), "fingerprint": after,
             "extra_columns_ignored": sorted(set(header) - set(REQUIRED_COLUMNS)),
             "excluded_rows": [], "negative_targets_retained": int((frame[TARGET] < 0).sum()),
             "zero_targets_retained": int((frame[TARGET] == 0).sum()),
             "identity_max_absolute_errors": errors,
             "row_id_policy": "A:sha256(schema_version,ordered_design_columns,normalized_coordinates)",
             "notes": ["Simulated cross-family test; not observed field validation",
                       "Integrity does not establish convergence or statistical independence"]}
    return frame, tuple("A:" + signature for signature in signatures), audit


def metrics(observed, prediction):
    observed, prediction = np.asarray(observed, dtype=float), np.asarray(prediction, dtype=float)
    if observed.ndim != 1 or prediction.shape != observed.shape or not np.isfinite(observed).all() or not np.isfinite(prediction).all():
        raise ValueError("Scoring requires aligned finite targets and predictions")
    if not len(observed):
        return {"n": 0, "R2": None, "RMSE": None, "MAE": None, "bias": None}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        error = prediction - observed
        result = {"n": len(observed), "R2": None if len(observed) < 2 or np.ptp(observed) == 0 else float(r2_score(observed, prediction)),
                  "RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(np.abs(error))),
                  "bias": float(error.mean())}
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError("Scoring produced nonfinite metrics")
    return result


def coverage(preprocessor, X, metadata, policy):
    """Transform with saved state only; thresholds never depend on A labels."""
    scaled = np.asarray(preprocessor.transform(X), dtype=float)
    if scaled.shape != (len(X), len(metadata["active_features"])) or not np.isfinite(scaled).all():
        raise ValueError("Saved preprocessing produced invalid coverage coordinates")
    raw = {name: ((X[name].to_numpy(dtype=float) < bounds["min"]) |
                  (X[name].to_numpy(dtype=float) > bounds["max"]))
           for name, bounds in metadata["observed_raw_ranges"].items()}
    scaled_flags = np.abs(scaled) > policy["standardized_threshold"]
    rows = []
    for position in range(len(X)):
        raw_names = [name for name, flags in raw.items() if flags[position]]
        scaled_names = [name for name, flag in zip(metadata["active_features"], scaled_flags[position]) if flag]
        rows.append({"ood_raw_range": bool(raw_names), "ood_scaled": bool(scaled_names),
                     "ood_any": bool(raw_names or scaled_names), "raw_range_features": raw_names,
                     "scaled_features": scaled_names})
    return rows, {"n": len(X), "raw_range_rows": sum(r["ood_raw_range"] for r in rows),
                  "scaled_rows": sum(r["ood_scaled"] for r in rows), "any_flag_rows": sum(r["ood_any"] for r in rows),
                  "raw_range_feature_counts": {name: int(flags.sum()) for name, flags in raw.items()},
                  "scaled_feature_counts": dict(zip(metadata["active_features"], scaled_flags.sum(axis=0).tolist()))}


def _table(path, rows, columns):
    temporary = path.with_name("." + path.name + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
    os.replace(temporary, path)


def evaluate(config, *, evaluation_dir=None):
    """Evaluate every frozen comparison; no model-scope/feature override or fallback.

    Load/verify frozen state BEFORE any A read. Independent event directories
    prevent historical evaluations from being overwritten by reruns.
    """
    config = config.resolved()
    if config.run_dir is None:
        raise ValueError("evaluate requires an explicit frozen --run-dir")
    if config.resume:
        raise ValueError("Evaluation cannot resume training; create a new evaluation event")
    root = Path(config.run_dir)
    frozen = load_frozen_models(root, device=config.device)
    if frozen.manifest["OOD_policy"] != OOD_POLICY:
        raise ValueError("Unsupported frozen OOD policy; evaluation cannot redefine thresholds")
    for metadata in frozen.metadata.values():
        if metadata["inference"]["OOD"] != frozen.manifest["OOD_policy"]:
            raise ValueError("Model and frozen OOD policies disagree")
    frozen_before = file_fingerprint(root / FROZEN_NAME)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if evaluation_dir is None:
        parent = root / "evaluations"
        if parent.is_symlink():
            raise ValueError("Symlinked evaluation parent is not allowed")
        parent.mkdir(exist_ok=True)
        directory = parent / f"A_{stamp}_{uuid.uuid4().hex[:8]}"
    else:
        from .config import PROJECT_ROOT
        supplied = Path(evaluation_dir).expanduser()
        directory = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
    if directory.resolve() != directory.absolute():
        raise ValueError("Symlinked evaluation output is not allowed")
    directory.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "status": "evaluating", "phase": PHASE,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
             "evaluation_dir": str(directory), "frozen_fingerprint": frozen_before,
             "frozen_manifest_checksum": frozen.manifest["manifest_checksum"],
             "primary_model_id": frozen.manifest["primary_model_id"], "model_ids": frozen.manifest["model_ids"],
             "requested_device": config.device, "workers": config.workers, "threads": config.threads,
             "execution": "sequential inference; no fitting worker pool",
             "model_policy": "all frozen comparisons; B-selected primary unchanged",
             "training_performed": False, "A_outcomes_accessed": False,
             "A_path": str(Path(config.test_csv).resolve()), "OOD_policy": frozen.manifest["OOD_policy"],
             "interpretation": "simulated cross-family A test; not observed field validation",
             "artifact_fingerprints": {}, "rejected_rows": []}
    state["evaluation_source_fingerprint"] = file_fingerprint(Path(__file__))
    atomic_json(directory / "evaluation_manifest.json", state)
    try:
        state["A_outcomes_accessed"] = True  # record attempted external loading even on failure
        frame, row_ids, audit = load_external(config.test_csv, frozen)
        with (root / "split_manifest.json").open(encoding="utf-8") as stream:
            B_signatures = {r["row_id"].split(":", 1)[1] for r in json.load(stream)["rows"]}
        audit["exact_design_overlap_with_B"] = sum(identity.split(":", 1)[1] in B_signatures for identity in row_ids)
        atomic_json(directory / "test_audit.json", audit)
        predictions, scores, subgroup_scores, summaries = [], [], [], {}
        for name in frozen.manifest["model_ids"]:
            model, metadata = frozen.models[name], frozen.metadata[name]
            preprocessing = metadata["preprocessing"]
            mode = preprocessing["specification"]["injection_mode"]
            # Explicit predictors/context only: labels and recovery totals never reach inference.
            X = frame.loc[:, list(FeatureSpec(mode).raw_features + CONTEXT_COLUMNS)].copy()
            with threadpool_limits(limits=1):
                flags, summaries[name] = coverage(fitted_preprocessor(model, metadata["kind"]), X,
                                                preprocessing, frozen.manifest["OOD_policy"])
                prediction = np.asarray(model.predict(X), dtype=float)
                raw = np.asarray(model.predict_raw(X), dtype=float) if metadata["inference"]["raw_prediction_available"] else None
            observed = frame[TARGET].to_numpy(dtype=float)
            outputs = [("primary", prediction)] + ([("raw", raw)] if raw is not None else [])
            base = {"model_id": name, "phase": PHASE, "is_B_selected_primary": name == state["primary_model_id"]}
            ood = np.asarray([r["ood_any"] for r in flags])
            for output, values in outputs:
                scores.append({**base, "output": output, "subgroup": "all", **metrics(observed, values)})
                for label, mask in (("no_OOD_flag", ~ood), ("any_OOD_flag", ood)):
                    subgroup_scores.append({**base, "output": output, "subgroup": label, **metrics(observed[mask], values[mask])})
            for position, identity in enumerate(row_ids):
                flag = flags[position]
                predictions.append({**base, "row_id": identity, "position": position, "csv_record_number": position + 1,
                                    "observed": float(observed[position]), "prediction": float(prediction[position]),
                                    "residual": float(observed[position] - prediction[position]),
                                    "raw_prediction": None if raw is None else float(raw[position]),
                                    "raw_residual": None if raw is None else float(observed[position] - raw[position]),
                                    **flag, "raw_range_features": json.dumps(flag["raw_range_features"]),
                                    "scaled_features": json.dumps(flag["scaled_features"])})
        if file_fingerprint(config.test_csv) != audit["fingerprint"]:
            raise ValueError("A CSV changed during evaluation")
        if file_fingerprint(root / FROZEN_NAME) != frozen_before:
            raise ValueError("Frozen completion marker changed during evaluation")
        for path, fingerprint in frozen.manifest["artifact_fingerprints"].items():
            if file_fingerprint(root / path) != fingerprint:
                raise ValueError("Frozen artifact changed during evaluation")
        _table(directory / "test_predictions.csv", predictions, PREDICTION_COLUMNS)
        _table(directory / "test_metrics.csv", scores, METRIC_COLUMNS)
        _table(directory / "test_subgroup_metrics.csv", subgroup_scores, METRIC_COLUMNS)
        atomic_json(directory / "coverage_diagnostics.json", {"policy": frozen.manifest["OOD_policy"], "models": summaries})
        files = ["test_predictions.csv", "test_metrics.csv", "test_subgroup_metrics.csv", "coverage_diagnostics.json", "test_audit.json"]
        state.update(status="evaluation_complete", A_rows=len(frame), A_fingerprint=audit["fingerprint"],
                     artifact_fingerprints={path: file_fingerprint(directory / path) for path in files})
        atomic_json(directory / "evaluation_manifest.json", state)
    except Exception as exc:
        state.update(status="evaluation_failed", error={"type": type(exc).__name__, "message": str(exc)},
                     rejected_rows=[{"scope": "entire_evaluation", "reason": str(exc),
                                     "policy": "No partial result accepted; no rows silently removed"}])
        atomic_json(directory / "evaluation_manifest.json", state)
        raise
    return {"status": "evaluation_complete", "evaluation_dir": str(directory), "A_rows": state["A_rows"],
            "primary_model_id": state["primary_model_id"], "model_ids": state["model_ids"],
            "training_performed": False, "interpretation": state["interpretation"]}

"""Verified saved-result reports and B outer-validation importance; no refitting."""

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .artifacts import atomic_json, canonical_hash, fitted_preprocessor, safe_artifact_path
from .config import PROJECT_ROOT
from .data import TARGET, file_fingerprint
from .evaluation import metrics
from .features import CONTEXT_COLUMNS, FeatureSpec, preprocessing_metadata
from .runtime import task_seed



def presentation_text(text):
    """Dataset names for human output; never applied to saved source artifacts."""
    aliases = {
        "B_nested_OOF_pooled": "Training/Validation (pooled out-of-fold)",
        "B_nested_OOF": "Training/Validation (out-of-fold)",
        "B_outer_validation": "Training/Validation (validation)",
        "B_holdout_validation": "Training/Validation (validation)",
        "A_external_simulation": "Testing", "B_validation": "Training/Validation (validation)",
        "A_simulation": "Testing", "B/A": "Training/Validation and Testing",
        "B-selected": "Training/Validation-selected", "all-B": "full Training/Validation",
        "B-validation": "validation", "B-training": "training",
        "B-based": "Training/Validation-based", "B-range": "Training/Validation-range",
    }
    for original, display in aliases.items():
        text = text.replace(original, display)
    return re.sub(r"\b[AB]\b", lambda match: {"B": "Training/Validation", "A": "Testing"}[match.group()], text)


def _report_table(frame):
    display = frame.copy()
    for column in ("phase", "split"):
        if column in display:
            display[column] = display[column].map(presentation_text)
    return display.to_string(index=False)


def validation_importance(estimator, heldout, *, kind, model_id, seed, fold, repeats=5):
    """Raw-column permutation on the actual development-fit model; B holdout only.

    RMSE increase in incremental percentage points; negative values are retained.
    Dependencies are recomputed by the unchanged saved preprocessing pipeline.
    """
    if not heldout["reservoir_family"].eq("B").all() or not heldout["source"].eq("OPM_SPE5_simulation").all():
        raise ValueError("Importance accepts B outer-validation rows only")
    metadata = preprocessing_metadata(fitted_preprocessor(estimator, kind))
    spec = FeatureSpec(metadata["specification"]["injection_mode"])
    X = heldout.loc[:, list(spec.raw_features + CONTEXT_COLUMNS)].copy()
    y = heldout[TARGET].to_numpy(dtype=float)
    records = []
    with threadpool_limits(limits=1):
        baseline_scores = metrics(y, estimator.predict(X))
        baseline = baseline_scores["RMSE"]
        for feature in spec.raw_features:
            for repeat in range(repeats):
                rng = np.random.default_rng(task_seed(seed, model_id, "B_importance:" + feature, fold, repeat))
                changed = X.copy()
                changed[feature] = X[feature].to_numpy()[rng.permutation(len(X))]
                permuted = metrics(y, estimator.predict(changed))
                increase = permuted["RMSE"] - baseline
                records.append({"model_id": model_id, "fold": fold, "feature": feature, "repeat": repeat,
                                "importance": increase, "importance_R2": None if baseline_scores["R2"] is None else baseline_scores["R2"] - permuted["R2"],
                                "n": len(X), "baseline_RMSE": baseline, "baseline_R2": baseline_scores["R2"]})
    return {"status": "complete", "scope": "B_outer_validation", "repeats": repeats,
            "validation_positions": heldout.index.tolist(),
            "definition": "RMSE(permuted)-RMSE(original) in percentage points; compatible R2(original)-R2(permuted) separately",
            "records": records, "notes": ["Marginal permutation; correlated inputs limit causal interpretation",
                                          "Constant raw predictors give zero importance; no A access"]}


def verify_figure_assets(root=PROJECT_ROOT):
    root = Path(root)
    with (root / "configs/figure_integrity.json").open(encoding="utf-8") as stream:
        expected = json.load(stream)["git_blob_hashes"]
    fingerprints = {}
    for relative, sha in expected.items():
        path = safe_artifact_path(root, relative)
        content = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != sha:
            raise ValueError(f"Preserved figure/font changed: {relative}")
        fingerprints[relative] = file_fingerprint(path)
    return fingerprints


def _read(root, name):
    with safe_artifact_path(root, name).open(encoding="utf-8") as stream:
        return json.load(stream)


def _verified_manifest(root, name, status):
    manifest = _read(root, name)
    if manifest.get("status") != status or not manifest.get("artifact_fingerprints"):
        raise ValueError("Reporting requires a completed result manifest")
    for relative, fingerprint in manifest["artifact_fingerprints"].items():
        if file_fingerprint(safe_artifact_path(root, relative)) != fingerprint:
            raise ValueError("Saved-result fingerprint mismatch")
    return manifest


def load_results(run_dir, evaluation_dir=None):
    """Read only completed, fingerprinted result files; never load model executables."""
    root = Path(run_dir).resolve()
    validation = _verified_manifest(root, "training_manifest.json", "validation_complete")
    if canonical_hash(validation["identity"]) != validation["identity_hash"]:
        raise ValueError("Validation identity checksum mismatch")
    selection = _read(root, "selection.json")
    if selection["primary_model_id"] != validation["primary_model_id"]:
        raise ValueError("Saved B winner metadata mismatch")
    protocol = selection["protocol"]
    prediction_file = "cv_oof_predictions.csv" if protocol == "nested-cv" else "validation_predictions.csv"
    predictions = pd.read_csv(safe_artifact_path(root, prediction_file))
    scores = pd.read_csv(safe_artifact_path(root, "validation_metrics.csv"))
    eligible = [r["model_id"] for r in selection["ranking"]]
    for name in eligible:
        rows = predictions[predictions.model_id == name]
        if rows.row_id.duplicated().any():
            raise ValueError("Duplicated validation prediction identities")
        computed = metrics(rows.observed, rows.prediction)
        saved = scores[scores.model_id == name]
        if len(saved) != 1 or not len(rows):
            raise ValueError("Incomplete validation metric inventory")
        _compare_metrics(computed, saved.iloc[0])
        if not np.allclose(rows.residual, rows.observed - rows.prediction):
            raise ValueError("Validation residual convention mismatch")
    frozen = None
    if (root / "frozen_manifest.json").exists():
        frozen = _verified_manifest(root, "frozen_manifest.json", "frozen")
        if canonical_hash({k: v for k, v in frozen.items() if k != "manifest_checksum"}) != frozen["manifest_checksum"]:
            raise ValueError("Frozen manifest checksum mismatch")
        if frozen["primary_model_id"] != selection["primary_model_id"]:
            raise ValueError("Frozen B winner mismatch")
    external = None
    if evaluation_dir is not None:
        directory = Path(evaluation_dir).resolve()
        event = _verified_manifest(directory, "evaluation_manifest.json", "evaluation_complete")
        if frozen is None or event["frozen_fingerprint"] != file_fingerprint(root / "frozen_manifest.json") or event["primary_model_id"] != selection["primary_model_id"] or event["model_ids"] != frozen["model_ids"]:
            raise ValueError("Evaluation does not belong to this frozen run")
        rows = pd.read_csv(safe_artifact_path(directory, "test_predictions.csv"))
        test_scores = pd.read_csv(safe_artifact_path(directory, "test_metrics.csv"))
        for name in frozen["model_ids"]:
            group = rows[rows.model_id == name]
            if len(group) != event["A_rows"] or group.row_id.duplicated().any():
                raise ValueError("Incomplete external prediction inventory")
            for _, score in test_scores[test_scores.model_id == name].iterrows():
                values = group.prediction if score.output == "primary" else group.raw_prediction
                _compare_metrics(metrics(group.observed, values), score)
        external = {"manifest": event, "predictions": rows, "metrics": test_scores,
                    "coverage": _read(directory, "coverage_diagnostics.json")}
    return {"root": root, "validation": validation, "selection": selection, "predictions": predictions,
            "metrics": scores, "tasks": _read(root, "task_records.json"), "frozen": frozen, "external": external}


def _compare_metrics(computed, saved):
    for name, value in computed.items():
        if value is None:
            if not pd.isna(saved[name]):
                raise ValueError("Undefined score was reported as defined")
        elif not np.isclose(value, saved[name], rtol=1e-9, atol=1e-10):
            raise ValueError("Saved metrics cannot be recomputed from predictions")


def _display(name, epochs=None):
    label = name.replace("_GUIDED", " (physics-guided)")
    if name.startswith("NN_"):
        label += f" ({epochs} epochs)" if epochs is not None else " (fold-specific epochs)"
    return label


def _legacy_label(name, epochs=None):
    if name.startswith("NN_"):
        return f"{name}({epochs})" if epochs is not None else f"{name}(fold-specific)"
    return name.replace("_GUIDED", "_PHYS")


def write_report(run_dir, *, evaluation_dir=None, report_dir=None):
    results = load_results(run_dir, evaluation_dir)
    root, selection = results["root"], results["selection"]
    if report_dir is None:
        parent = root / "reports"
        if parent.is_symlink():
            raise ValueError("Symlinked report parent is not allowed")
        parent.mkdir(exist_ok=True)
        directory = parent / (datetime.now(timezone.utc).strftime("report_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    else:
        supplied = Path(report_dir).expanduser()
        directory = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
    if directory.absolute() != directory.resolve():
        raise ValueError("Symlinked report output is not allowed")
    directory.mkdir(parents=True, exist_ok=False)
    primary = selection["primary_model_id"]
    protocol = selection["protocol"]
    eligible = [r["model_id"] for r in selection["ranking"]]
    compatible, labels = [], []
    for name in eligible:
        task_epochs = [r.get("selected_epochs") for r in results["tasks"] if r["model_id"] == name]
        epochs = task_epochs[0] if len(set(task_epochs)) == 1 else None
        labels.append({"model_id": name, "display_label": _display(name, epochs), "legacy_model": _legacy_label(name, epochs)})
        row = results["metrics"][results["metrics"].model_id == name].iloc[0]
        compatible.append({"split": "B_nested_OOF_pooled" if protocol == "nested-cv" else "B_holdout_validation",
                           "model": _legacy_label(name, epochs), "model_id": name, "epochs": epochs, "mode": "physics-guided" if name.endswith("_GUIDED") or name == "NN_PINN_HARD" else "plain",
                           **{k: row[k] for k in ("R2", "RMSE", "MAE")}})
    if results["external"] is not None:
        for _, row in results["external"]["metrics"].iterrows():
            if row.output == "primary":
                entry = results["frozen"]["models"][row.model_id]
                metadata = _read(root, entry["files"]["metadata"])
                compatible.append({"split": "A_external_simulation", "model": _legacy_label(row.model_id, metadata["configuration"].get("selected_epochs")), "model_id": row.model_id,
                                   "epochs": metadata["configuration"].get("selected_epochs"),
                                   "mode": "saved_inference_policy", **{k: row[k] for k in ("R2", "RMSE", "MAE")}})
    exported = pd.DataFrame(compatible, columns=["split", "model", "model_id", "epochs", "mode", "R2", "RMSE", "MAE"])
    exported["dataset_label"] = exported["split"].map(lambda split: "Testing" if split == "A_external_simulation" else "Training/Validation")
    exported.to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(labels).to_csv(directory / "model_labels.csv", index=False)
    skipped = ["Legacy split-sensitivity Figures 5/9: unavailable; the opt-in manuscript set substitutes B/A errors and A applicability diagnostics",
               "Raw-data Figures 1/2 require plots --figure-set manuscript and the original fingerprint-bound CSVs"]
    if protocol == "nested-cv":
        cv = results["metrics"].copy()
        aliases = {row["model_id"]: row["legacy_model"] for row in labels}
        pd.DataFrame({"model": cv.model_id.map(aliases), "model_id": cv.model_id, "R2": cv.mean_outer_R2, "RMSE": cv.mean_outer_RMSE, "MAE": cv.mean_outer_MAE}).to_csv(directory / "cv_metrics.csv", index=False)
        if results["validation"]["config"]["outer_folds"] != 5:
            skipped.append("Legacy Figure 3: its fixed five-fold label does not match this run's outer-fold count")
    else:
        skipped.append("Figure 3/CV export: holdout validation is not cross-validation")
    importance_tasks = [r for r in results["tasks"] if r["model_id"] == primary]
    importance_records = [record for r in importance_tasks for record in r.get("importance", {}).get("records", [])]
    if importance_tasks and all(r.get("importance", {}).get("status") == "complete" for r in importance_tasks):
        for task in importance_tasks:
            diagnostic = task["importance"]
            positions = [row["position"] for row in task["predictions"]]
            if diagnostic.get("scope") != "B_outer_validation" or diagnostic.get("validation_positions") != positions or any(
                r["model_id"] != primary or r["fold"] != task["fold"] or r["n"] != len(positions) for r in diagnostic["records"]):
                raise ValueError("Importance does not match B outer-validation identities")
        records = pd.DataFrame(importance_records)
        records.to_csv(directory / "importance_records.csv", index=False)
        summary = records.groupby("feature", sort=False).importance.agg(["mean", "std"]).reset_index()
        summary.columns = ["feature", "importance_mean", "importance_std"]
        summary.to_csv(directory / "feature_importance_rmse.csv", index=False)
        if "importance_R2" in records and records.importance_R2.notna().all():
            r2 = records.groupby("feature", sort=False).importance_R2.agg(["mean", "std"]).reset_index()
            r2.columns = ["feature", "importance_mean", "importance_std"]
            r2.to_csv(directory / "feature_importance.csv", index=False)
        else:
            skipped.append("Legacy Figure 8/R2 importance: R2 undefined or not recorded; RMSE diagnostics remain separately available")
    else:
        skipped.append("Importance: B development-model diagnostics unavailable or failed; no final-all-B substitution")
    lines = ["WAG experiment report", f"Primary model selected using Training/Validation: {primary}", f"Selection rule: {selection['rule']}",
             f"B rows: {results['validation']['B_rows']}; protocol: {protocol}",
             "The validation subset of Training/Validation is used for model selection; it is not an unbiased post-selection performance estimate.",
             "Incremental recovery is WAG minus waterflood, in percentage points of initial oil in place.",
             "Residual = observed - predicted; bias = mean(predicted - observed).",
             "Training/Validation metrics (validation predictions):", _report_table(results["metrics"])]
    if protocol == "nested-cv":
        lines.append("Pooled OOF metrics differ from mean/std outer-fold metrics; selection uses mean outer RMSE.")
    if results["frozen"] is not None:
        lines += [f"Final refit: {results['frozen']['B']['rows']} B rows; frozen models: {', '.join(results['frozen']['model_ids'])}.",
                  "Final fitting is a separate stage; training predictions are not validation scores."]
    else:
        lines.append("Final refit/freeze: not available.")
    if results["external"] is not None:
        lines += ["Testing metrics:", _report_table(results["external"]["metrics"]),
                  "Testing uses simulated data, not observed field validation. The primary selected using Training/Validation stays unchanged.",
                  "Coverage diagnoses marginal departures from Training/Validation ranges and scaling; no row exclusion or joint-support guarantee.",
                  json.dumps(results["external"]["coverage"], indent=2)]
    else:
        lines.append("A evaluation: not supplied; no test performance inferred.")
    lines += ["Importance uses held-out validation predictions from each development-fit model; five permutations per raw feature.",
              "Dependencies are recomputed. Correlated features limit causal interpretation; negative importance is retained.",
              "Importance mean/std summarize fold/permutation repeats, not confidence intervals or causal effects.",
              "feature_importance.csv is compatible R2 decrease; feature_importance_rmse.csv is RMSE increase in percentage points.",
              "No bootstrap intervals, PDE-residual guarantee or observed-field accuracy is claimed.",
              "Skipped/unavailable:", *skipped, "Excluded comparisons:", json.dumps(selection["excluded_models"], indent=2)]
    for task in importance_tasks:
        if task.get("importance", {}).get("status") != "complete":
            lines.append(f"Importance fold {task['fold']}: " + json.dumps(task.get("importance", {"status": "not recorded"})))
    (directory / "report.txt").write_text(presentation_text("\n".join(lines) + "\n"), encoding="utf-8")
    files = [p.name for p in directory.iterdir() if p.is_file()]
    manifest = {"schema_version": 1, "status": "report_complete", "run_dir": str(root), "primary_model_id": primary,
                "training_performed": False, "skipped": skipped, "validation_fingerprint": file_fingerprint(root / "training_manifest.json"),
                "evaluation_fingerprint": None if evaluation_dir is None else file_fingerprint(Path(evaluation_dir) / "evaluation_manifest.json"),
                "artifact_fingerprints": {name: file_fingerprint(directory / name) for name in files}}
    atomic_json(directory / "report_manifest.json", manifest)
    return {"status": "report_complete", "report_dir": str(directory), "primary_model_id": primary, "training_performed": False}

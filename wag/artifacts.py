"""Atomic trusted model bundles and verified frozen-model loading; no fitting.

Imports do not access datasets/files, create workers or initialize CUDA.
Joblib bundles are executable Python artifacts; load only trusted run directories.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import file_fingerprint
from .features import CONTEXT_COLUMNS, FeatureSpec, preprocessing_metadata
from .models.classical import assert_cpu_estimator


ARTIFACT_VERSION = 1
FROZEN_NAME = "frozen_manifest.json"
OOD_POLICY = {"action": "diagnose_only", "exclude_rows": False, "prediction_clamp": False,
              "raw_range": "any canonical raw input outside saved training min/max",
              "standardized_features": "any absolute saved-scaled active feature exceeds threshold",
              "standardized_threshold": 3.0}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Symlinked artifact output is not allowed")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_binary(path, writer):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Symlinked model output is not allowed")
    writer(temporary)
    os.replace(temporary, path)


def safe_artifact_path(directory, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Artifact path must remain inside the run directory")
    root = Path(directory).resolve()
    candidate = root / relative
    if candidate.resolve() != candidate or not candidate.is_file():
        raise ValueError("Artifact is missing or symlinked")
    return candidate


def fitted_preprocessor(estimator, kind):
    if kind == "neural":
        return estimator.preprocessor
    if hasattr(estimator, "preprocessor_"):
        return estimator.preprocessor_
    return estimator.named_steps["preprocess"]


def prediction_probe(development, mode):
    """Only B predictors/context; neither labels nor recovery totals are serialized."""
    names = list(FeatureSpec(mode).raw_features + CONTEXT_COLUMNS)
    X = development.frame.iloc[:min(16, len(development.frame))][names]
    return {"row_ids": list(development.row_ids[:len(X)]), "columns": names,
            "inputs": json.loads(X.to_json(orient="records", double_precision=15))}


def save_model_bundle(directory, model_id, kind, estimator, configuration, fit_record, probe):
    directory = Path(directory)
    stem = "final_models/" + hashlib.sha256(model_id.encode()).hexdigest()
    preprocessor = fitted_preprocessor(estimator, kind)
    preprocessing = preprocessing_metadata(preprocessor)
    guided = kind == "classical" and hasattr(estimator, "predict_raw")
    bounded = kind == "neural" and model_id == "NN_PINN_HARD"
    bounds = list(estimator.bounds) if guided else (configuration["settings"]["bounds"] if bounded else None)
    information = {"version": ARTIFACT_VERSION, "model_id": model_id, "kind": kind,
                   "configuration": deepcopy(configuration), "configuration_hash": canonical_hash(configuration),
                   "fit_record": deepcopy(fit_record), "preprocessing": preprocessing,
                   "inference": {"policy": "guided_clip" if guided else "architectural_bounds" if bounded else "unbounded",
                                 "bounds": bounds, "raw_prediction_available": guided, "OOD": deepcopy(OOD_POLICY)}}
    files = {}
    if kind == "classical":
        path = stem + ".joblib"
        _atomic_binary(directory / path, lambda p: joblib.dump(estimator, p, compress=3))
        files["model"] = path
    elif kind == "neural":
        import torch
        state, preprocessing_path = stem + ".pt", stem + ".preprocessing.joblib"
        _atomic_binary(directory / state, lambda p: torch.save(estimator.state_bundle(), p))
        _atomic_binary(directory / preprocessing_path, lambda p: joblib.dump(preprocessor, p, compress=3))
        files.update(state=state, preprocessor=preprocessing_path)
    else:
        raise ValueError("Unknown model artifact kind")
    metadata_path = stem + ".metadata.json"
    atomic_json(directory / metadata_path, information)
    files["metadata"] = metadata_path
    entry = {"model_id": model_id, "kind": kind, "files": files,
             "fingerprints": {path: file_fingerprint(directory / path) for path in files.values()},
             "configuration_hash": information["configuration_hash"]}
    # This verifies a CPU reload against the live model, before any freeze marker.
    restored = load_model_bundle(directory, entry, device="cpu")
    X = pd.DataFrame(probe["inputs"], columns=probe["columns"])
    with threadpool_limits(limits=1):
        expected, actual = np.asarray(estimator.predict(X)), np.asarray(restored.predict(X))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
        verification = {"row_ids": probe["row_ids"], "prediction": expected.tolist(), "rtol": 1e-5, "atol": 1e-4}
        if guided:
            expected_raw = np.asarray(estimator.predict_raw(X))
            np.testing.assert_allclose(restored.predict_raw(X), expected_raw, rtol=1e-5, atol=1e-4)
            verification["raw_prediction"] = expected_raw.tolist()
    if not np.isfinite(actual).all():
        raise ValueError("Serialized prediction check produced nonfinite values")
    entry["verification"] = verification
    return entry


def load_model_bundle(directory, entry, *, device="cpu"):
    """Verify checksums before deserialization; CPU is the portable default."""
    if not isinstance(entry, dict) or entry.get("kind") not in ("classical", "neural") or not isinstance(entry.get("files"), dict):
        raise ValueError("Invalid model artifact entry")
    files = entry["files"]
    expected_roles = {"model", "metadata"} if entry["kind"] == "classical" else {"state", "preprocessor", "metadata"}
    if set(files) != expected_roles or set(entry["fingerprints"]) != set(files.values()):
        raise ValueError("Incomplete model file inventory")
    for path, fingerprint in entry["fingerprints"].items():
        if file_fingerprint(safe_artifact_path(directory, path)) != fingerprint:
            raise ValueError("Model artifact fingerprint mismatch")
    with safe_artifact_path(directory, files["metadata"]).open(encoding="utf-8") as stream:
        information = json.load(stream)
    if information.get("version") != ARTIFACT_VERSION or information.get("model_id") != entry["model_id"] or information.get("kind") != entry["kind"] or canonical_hash(information["configuration"]) != entry["configuration_hash"] or information.get("configuration_hash") != entry["configuration_hash"]:
        raise ValueError("Model metadata/configuration mismatch")
    if entry["kind"] == "classical":
        estimator = joblib.load(safe_artifact_path(directory, files["model"]))
        assert_cpu_estimator(estimator)
        if hasattr(estimator, "estimator_"):
            assert_cpu_estimator(estimator.estimator_)
    else:
        from .runtime import configure_torch, dependency_preflight
        dependency_preflight(["torch"])
        configure_torch(device)
        import torch
        from .neural import load_neural_bundle
        bundle = torch.load(safe_artifact_path(directory, files["state"]), map_location="cpu", weights_only=True)
        estimator = load_neural_bundle(bundle, joblib.load(safe_artifact_path(directory, files["preprocessor"])), device=device)
    if preprocessing_metadata(fitted_preprocessor(estimator, entry["kind"])) != information["preprocessing"]:
        raise ValueError("Executable preprocessing differs from saved metadata")
    return estimator


@dataclass
class FrozenModels:
    manifest: dict
    models: dict
    metadata: dict


def load_frozen_models(directory, *, device="cpu"):
    """No fitting/data access; reject incomplete or tampered artifacts before loading."""
    directory = Path(directory).resolve()
    with safe_artifact_path(directory, FROZEN_NAME).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    return _verify_and_load(directory, manifest, device=device)


def publish_frozen_manifest(directory, manifest):
    """Check the complete candidate before atomically creating the freeze marker."""
    directory = Path(directory).resolve()
    if (directory / FROZEN_NAME).exists():
        raise FileExistsError("A frozen manifest is immutable; it cannot be overwritten")
    manifest = deepcopy(manifest)
    manifest["manifest_checksum"] = canonical_hash(manifest)
    verified = _verify_and_load(directory, manifest, device="cpu")
    atomic_json(directory / FROZEN_NAME, manifest)
    return verified


def _verify_and_load(directory, manifest, *, device):
    try:
        if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_VERSION or manifest.get("status") != "frozen" or manifest.get("models_frozen") is not True:
            raise ValueError("Run does not contain a valid frozen-model completion marker")
        body = {key: value for key, value in manifest.items() if key != "manifest_checksum"}
        if canonical_hash(body) != manifest.get("manifest_checksum"):
            raise ValueError("Frozen manifest checksum mismatch")
        if not manifest["model_ids"] or set(manifest["model_ids"]) != set(manifest["models"]) or len(set(manifest["model_ids"])) != len(manifest["model_ids"]) or manifest["primary_model_id"] not in manifest["models"]:
            raise ValueError("Frozen winner/model inventory mismatch")
        if type(manifest["B"]["rows"]) is not int or manifest["B"]["rows"] < 1 or manifest.get("A_outcomes_accessed") is not False:
            raise ValueError("Frozen dataset-role/count metadata mismatch")
        required = {"training_manifest.json", "selection.json", "task_records.json", "split_manifest.json", "data_audit.json",
                    "refit_manifest.json", "refit_plan.json", "refit_records.json", "refit_probe.json"}
        if not required <= set(manifest["artifact_fingerprints"]):
            raise ValueError("Frozen provenance file inventory is incomplete")
        for path, fingerprint in manifest["artifact_fingerprints"].items():
            if file_fingerprint(safe_artifact_path(directory, path)) != fingerprint:
                raise ValueError("Frozen artifact fingerprint mismatch")
        if manifest["probe_file"] not in manifest["artifact_fingerprints"]:
            raise ValueError("Frozen prediction probe is not fingerprinted")
        def read(name):
            with safe_artifact_path(directory, name).open(encoding="utf-8") as stream:
                return json.load(stream)
        refit, validation, selection, audit = (read(name) for name in
            ("refit_manifest.json", "training_manifest.json", "selection.json", "data_audit.json"))
        if (refit["status"] != "refit_complete" or canonical_hash(refit["identity"]) != manifest["refit_identity_hash"]
                or refit["identity_hash"] != manifest["refit_identity_hash"] or refit["model_ids"] != manifest["model_ids"]
                or validation["status"] != "validation_complete" or validation["identity_hash"] != manifest["validation_identity_hash"]
                or selection["primary_model_id"] != manifest["primary_model_id"]
                or audit["rows"] != manifest["B"]["rows"] or audit["fingerprint"] != manifest["B"]["fingerprint"]):
            raise ValueError("Frozen selection/refit/B provenance mismatch")
        from .runtime import dependency_preflight
        # Serialized sklearn objects require the same library versions; torch CPU
        # and CUDA build suffixes may differ, while the release must match.
        needed = {"numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"}
        if any(entry["kind"] == "neural" for entry in manifest["models"].values()):
            needed.add("torch")
        if any(name in manifest["models"] for name in ("XGBoost", "XGBoost_GUIDED")):
            needed.add("xgboost")
        if any(name in manifest["models"] for name in ("LightGBM", "LightGBM_GUIDED")):
            needed.add("lightgbm")
        if not needed <= set(manifest["packages"]):
            raise ValueError("Frozen package inventory is incomplete")
        packages = sorted(needed)
        current = dependency_preflight(packages)
        for package, version in current.items():
            saved = manifest["packages"][package]
            compatible = version.split("+")[0] == saved.split("+")[0] if package == "torch" else version == saved
            if not compatible:
                raise ValueError(f"Frozen loading requires compatible saved package versions: {package}")
        for entry in manifest["models"].values():
            for path, fingerprint in entry["fingerprints"].items():
                if manifest["artifact_fingerprints"].get(path) != fingerprint:
                    raise ValueError("Frozen model inventory is not fully fingerprinted")
        probe_path = safe_artifact_path(directory, manifest["probe_file"])
        with probe_path.open(encoding="utf-8") as stream:
            probe = json.load(stream)
        X = pd.DataFrame(probe["inputs"], columns=probe["columns"])
        models, information = {}, {}
        for name in manifest["model_ids"]:
            entry = manifest["models"][name]
            if entry["model_id"] != name or entry["verification"]["row_ids"] != probe["row_ids"]:
                raise ValueError("Frozen model/probe row identities mismatch")
            models[name] = load_model_bundle(directory, entry, device=device)
            with threadpool_limits(limits=1):
                verify = entry["verification"]
                np.testing.assert_allclose(models[name].predict(X), verify["prediction"], rtol=verify["rtol"], atol=verify["atol"])
                if "raw_prediction" in verify:
                    np.testing.assert_allclose(models[name].predict_raw(X), verify["raw_prediction"], rtol=verify["rtol"], atol=verify["atol"])
            with safe_artifact_path(directory, entry["files"]["metadata"]).open(encoding="utf-8") as stream:
                information[name] = json.load(stream)
            if information[name]["preprocessing"]["fit_rows"] != manifest["B"]["rows"]:
                raise ValueError("Frozen preprocessing was not fitted on all B")
        return FrozenModels(manifest, models, information)
    except (KeyError, TypeError, AttributeError, AssertionError, IndexError) as exc:
        raise ValueError(f"Invalid frozen artifact state: {exc}") from exc

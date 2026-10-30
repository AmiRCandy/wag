"""Resumable full experiment: B development/freeze, A test, saved-result reports.

Imports are standard-library only. No datasets, numeric libraries, CUDA or
workers are accessed until run_pipeline is explicitly called.
"""

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from .config import PROJECT_ROOT


@contextmanager
def _lock(root):
    # A sibling lock permits run_training to create its own NEW run directory.
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / ("." + root.name + ".pipeline.lock")
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError("Pipeline is locked; verify no active process before manually removing its stale sibling .pipeline.lock") from exc
    try:
        yield
    finally:
        path.unlink()


def run_pipeline(config, *, runtime=None, no_plots=False, _asset_root=PROJECT_ROOT):
    """One explicit run, with no A access before successful B model freezing.

    Explicit resume verifies checkpoints; successful evaluation is not silently
    repeated. Failed event directories remain separate historical attempts.
    """
    from .runtime import THREAD_ENV, dependency_preflight
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("Call bootstrap_threads once before numeric imports; full workflow requires one CPU thread")
    from .artifacts import atomic_json, canonical_hash, load_frozen_models
    from .data import file_fingerprint
    from .evaluation import evaluate
    from .reporting import _verified_manifest, load_results, verify_figure_assets, write_report
    from .training import final_refit, run_training
    if type(no_plots) is not bool:
        raise ValueError("no_plots must be boolean")
    config = config.resolved()
    if config.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config = replace(config, run_dir=str(PROJECT_ROOT / "results/runs" / f"run_{stamp}_{uuid.uuid4().hex[:8]}"))
    root = Path(config.run_dir)
    if not config.resume and root.exists():
        raise FileExistsError("run requires a NEW directory; use explicit --resume for a matching pipeline")
    if config.resume and not root.is_dir():
        raise ValueError("Pipeline resume requires an existing run directory")
    if not no_plots:
        dependency_preflight(["matplotlib", "seaborn"])
        verify_figure_assets(_asset_root)  # fail before expensive fitting; never opens A
    values = asdict(config)
    values.pop("resume")
    identity = {"config": values, "no_plots": no_plots,
                "sources": {str(path.relative_to(PROJECT_ROOT)): file_fingerprint(path)
                            for path in sorted((PROJECT_ROOT / "wag").rglob("*.py"))}}
    identity["sources"]["pyproject.toml"] = file_fingerprint(PROJECT_ROOT / "pyproject.toml")
    setting_paths = []
    if config.model_scope in ("all", "classical"):
        setting_paths.append(config.classical_config)
    if config.model_scope in ("all", "neural"):
        setting_paths.append(config.neural_config)
    identity["settings"] = {name: file_fingerprint(name) for name in setting_paths}
    identity_hash = canonical_hash(identity)
    path = root / "pipeline_manifest.json"

    def save(state):
        payload = dict(state)
        payload["checksum"] = canonical_hash(state)
        atomic_json(path, payload)

    def read_state():
        if path.is_symlink():
            raise ValueError("Symlinked pipeline checkpoint is not allowed")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Invalid pipeline manifest schema")
        state = {k: v for k, v in payload.items() if k != "checksum"}
        if payload.get("checksum") != canonical_hash(state) or state.get("identity_hash") != identity_hash or state.get("identity") != identity:
            raise ValueError("Pipeline resume requires identical configuration/code/plot policy and a valid checksum")
        if state.get("schema_version") != 1 or not isinstance(state.get("stages"), dict) or not isinstance(state.get("events"), list):
            raise ValueError("Invalid pipeline checkpoint schema")
        if state.get("status") not in ("running", "pipeline_complete", "pipeline_failed", "pipeline_incomplete") or any(not isinstance(record, dict) for record in state["stages"].values()):
            raise ValueError("Invalid pipeline stage records")
        return state

    def verify_report(directory, event):
        manifest = _verified_manifest(directory, "report_manifest.json", "report_complete")
        if manifest["validation_fingerprint"] != file_fingerprint(root / "training_manifest.json") or manifest["evaluation_fingerprint"] != file_fingerprint(event / "evaluation_manifest.json"):
            raise ValueError("Report checkpoint does not match pipeline inputs")
        if not no_plots:
            _verified_manifest(directory, "plot_manifest.json", "plots_complete")

    def summary(state):
        return {"status": state["status"], "run_dir": str(root), "primary_model_id": state.get("primary_model_id"),
                "evaluation_dir": state["stages"].get("evaluation", {}).get("evaluation_dir"),
                "report_dir": state["stages"].get("report", {}).get("report_dir"),
                "models_frozen": (root / "frozen_manifest.json").exists(), "plots_requested": not no_plots}

    with _lock(root):
        if not config.resume and root.exists():
            raise FileExistsError("Run directory appeared concurrently; no overwrite permitted")
        state = read_state() if config.resume else {"schema_version": 1, "status": "running", "identity": identity,
                "identity_hash": identity_hash, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "stages": {}, "events": [], "A_outcomes_accessed": False}
        stage = "B_validation"
        try:
            if (root / "frozen_manifest.json").exists():
                recorded = state["stages"].get("freeze", {}).get("fingerprint")
                if recorded is not None and file_fingerprint(root / "frozen_manifest.json") != recorded:
                    raise ValueError("Frozen checkpoint changed after pipeline recording")
                frozen = load_frozen_models(root, device="cpu")
                saved_config = dict(frozen.manifest["config"])
                for key in ("resume", "test_csv"):
                    saved_config.pop(key, None)
                requested = asdict(config)
                for key in ("resume", "test_csv"):
                    requested.pop(key, None)
                if saved_config != requested:
                    raise ValueError("Frozen run configuration differs from pipeline request")
                if Path(config.train_csv).exists() and file_fingerprint(config.train_csv) != frozen.manifest["B"]["fingerprint"]:
                    raise ValueError("B CSV changed since freezing")
                state["primary_model_id"] = frozen.manifest["primary_model_id"]
            else:
                if not (root / "refit_manifest.json").exists():
                    validation = run_training(config, runtime=runtime)
                    state["stages"]["validation"] = validation
                    save(state)
                    if validation["status"] != "validation_complete":
                        state.update(status="pipeline_incomplete", failed_stage=stage)
                        save(state)
                        return summary(state)
                stage = "B_final_refit"
                fitting = final_refit(config)
                state["stages"]["refit"] = fitting
                save(state)
                if fitting["status"] != "frozen":
                    state.update(status="pipeline_incomplete", failed_stage=stage)
                    save(state)
                    return summary(state)
                state["primary_model_id"] = fitting["primary_model_id"]
            state["stages"]["freeze"] = {"status": "frozen", "fingerprint": file_fingerprint(root / "frozen_manifest.json")}
            stage = "A_evaluation"
            event_record = state["stages"].get("evaluation", {})
            event = Path(event_record["evaluation_dir"]) if event_record.get("evaluation_dir") else None
            event_complete = False
            if event is not None and (event / "evaluation_manifest.json").exists():
                with (event / "evaluation_manifest.json").open(encoding="utf-8") as stream:
                    event_complete = json.load(stream).get("status") == "evaluation_complete"
            if event_complete:
                if event_record.get("fingerprint") is not None and file_fingerprint(event / "evaluation_manifest.json") != event_record["fingerprint"]:
                    raise ValueError("Recorded evaluation checkpoint changed")
                results = load_results(root, event)
                fingerprint = results["external"]["manifest"]["A_fingerprint"]
                if Path(config.test_csv).exists() and file_fingerprint(config.test_csv) != fingerprint:
                    raise ValueError("A CSV changed after recorded testing; use evaluate explicitly for a new event")
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                event = root / "evaluations" / f"pipeline_A_{stamp}_{uuid.uuid4().hex[:8]}"
                state["stages"]["evaluation"] = {"status": "running", "evaluation_dir": str(event)}
                state["events"].append({"stage": "evaluation", "directory": str(event)})
                state["A_outcomes_accessed"] = True  # conservative attempted-access record, after freezing only
                save(state)
                evaluation = evaluate(replace(config, resume=False), evaluation_dir=event)
                state["stages"]["evaluation"] = evaluation
            state["A_outcomes_accessed"] = True
            state["stages"]["evaluation"]["status"] = "evaluation_complete"
            state["stages"]["evaluation"]["fingerprint"] = file_fingerprint(event / "evaluation_manifest.json")
            stage = "report_plots"
            report_record = state["stages"].get("report", {})
            report = Path(report_record["report_dir"]) if report_record.get("report_dir") else None
            report_complete = False
            if report is not None and (report / "report_manifest.json").exists():
                if no_plots:
                    report_complete = True
                elif (report / "plot_manifest.json").exists():
                    with (report / "plot_manifest.json").open(encoding="utf-8") as stream:
                        report_complete = json.load(stream).get("status") == "plots_complete"
            if report_complete:
                if report_record.get("fingerprint") is not None and file_fingerprint(report / "report_manifest.json") != report_record["fingerprint"]:
                    raise ValueError("Recorded report checkpoint changed")
                verify_report(report, event)
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                report = root / "reports" / f"pipeline_report_{stamp}_{uuid.uuid4().hex[:8]}"
                state["stages"]["report"] = {"status": "running", "report_dir": str(report)}
                state["events"].append({"stage": "report", "directory": str(report)})
                save(state)
                if no_plots:
                    output = write_report(root, evaluation_dir=event, report_dir=report)
                else:
                    from .plotting import plot_saved_results
                    output = plot_saved_results(root, evaluation_dir=event, report_dir=report, _asset_root=_asset_root)
                state["stages"]["report"] = output
                verify_report(report, event)
            state["stages"]["report"]["fingerprint"] = file_fingerprint(report / "report_manifest.json")
            if state["status"] != "pipeline_complete":
                state.update(status="pipeline_complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
                state.pop("error", None)
                state.pop("failed_stage", None)
                save(state)
            return summary(state)
        except BaseException as exc:
            # Never write into a directory which another caller created first.
            if root.is_dir() and not (isinstance(exc, FileExistsError) and not config.resume and not path.exists()):
                state.update(status="pipeline_failed", failed_stage=stage,
                             error={"type": type(exc).__name__, "message": str(exc)})
                save(state)
            raise

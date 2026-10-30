"""CLI parsing is safe without installed machine-learning dependencies."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import sys
import uuid

from .config import PROJECT_ROOT, load_config


def parser():
    result = argparse.ArgumentParser(description="WAG pipeline: B development, frozen A test, verified reports and plots")
    commands = result.add_subparsers(dest="command", required=True)
    for command, help_text in (("run", "Develop/refit on B, then evaluate A"),
                               ("train", "B-only validation, final all-B refit and model freeze"),
                               ("refit", "Finalize a completed B validation run without repeating validation"),
                               ("evaluate", "Evaluate frozen models on A without fitting"),
                               ("audit", "Audit B without scoring A"),
                               ("report", "Report verified saved validation/test results without fitting"),
                               ("plots", "Plot saved results without fitting")):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--config", help="JSON configuration path, relative to repository root")
        sub.add_argument("--classical-config", help="Classical registry/search JSON, relative to repository root")
        sub.add_argument("--neural-config", help="Neural architecture/constraint JSON, relative to repository root")
        sub.add_argument("--model-scope", choices=("all", "classical", "neural"), help="Explicit experiment scope; default all")
        sub.add_argument("--resume", action="store_true", default=None, help="Run/train/refit: resume matching checkpoints in --run-dir")
        if command == "run":
            sub.add_argument("--no-plots", action="store_true", help="Explicit report-only full workflow; omit image rendering")
        if command == "train":
            sub.add_argument("--validation-only", action="store_true", help="Stop after B validation/selection; do not refit/freeze")
        if command == "evaluate":
            sub.add_argument("--evaluation-dir", help="Optional NEW root-relative event directory; default unique directory under --run-dir/evaluations")
        if command in ("report", "plots"):
            sub.add_argument("--evaluation-dir", help="Explicit completed A event to include; omitted means B-only report")
            sub.add_argument("--report-dir", help="Optional NEW report output directory")
        if command == "plots":
            sub.add_argument("--figure-set", choices=("diagnostics", "manuscript"), default="diagnostics", help="manuscript: nine numbered figures for a completed 80/20 B run and A event; reads bound CSVs, never refits")
            sub.add_argument("--fig1-label-size", type=float, default=24, help="Figure 1 x/y axis label font size in points (default: 24)")
            sub.add_argument("--fig1-tick-size", type=float, default=16, help="Figure 1 tick and scientific multiplier font size in points (default: 16)")
        sub.add_argument("--device", help="cpu or cuda:N (logical visible-device index)")
        sub.add_argument("--workers", type=int, help="1..8 CPU fitting workers (default: 8)")
        sub.add_argument("--threads", type=int, help="Must be 1")
        sub.add_argument("--seed", type=int)
        sub.add_argument("--validation", choices=("holdout", "nested-cv"))
        sub.add_argument("--injection-mode", choices=("volumes", "design", "rates"))
        sub.add_argument("--validation-fraction", type=float)
        sub.add_argument("--inner-folds", type=int)
        sub.add_argument("--outer-folds", type=int)
        sub.add_argument("--train-csv")
        sub.add_argument("--test-csv")
        sub.add_argument("--run-dir", help="Root-relative run path; runtime checks require a NEW directory")
        sub.add_argument("--check-runtime", action="store_true",
                         help="Validate runtime only; print manifest or save it to a NEW --run-dir. No data access/fitting.")
        if command == "audit":
            sub.add_argument("--check-test-inputs", action="store_true",
                             help="Opt-in A predictor-only schema/exact-overlap diagnostic; never load A outcomes")
    return result


def audit_command(config, manifest, check_inputs):
    from .data import (check_test_inputs, load_development, make_split_manifest,
                       save_audit_artifacts)
    development = load_development(config.train_csv, test_path=config.test_csv)
    splits = make_split_manifest(development, config)
    audit = {"development": development.audit, "test_inputs": None}
    if check_inputs:
        audit["test_inputs"] = check_test_inputs(development, config.test_csv,
                                                 injection_mode=config.injection_mode)
    if config.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config = replace(config, run_dir=str(PROJECT_ROOT / "results" / "runs" /
                                             f"audit_{stamp}_{uuid.uuid4().hex[:8]}"))
    from dataclasses import asdict
    manifest.update({"status": "audit_complete", "config": asdict(config),
                     "data_accessed": True, "A_outcomes_accessed": False,
                     "B_fingerprint": development.fingerprint,
                     "A_predictors_accessed": check_inputs,
                     "A_fingerprint": audit["test_inputs"]["fingerprint"] if check_inputs else None,
                     "artifacts": ["data_audit.json", "split_manifest.json"],
                     "created_at_utc": datetime.now(timezone.utc).isoformat()})
    directory = save_audit_artifacts(config.run_dir, audit, splits, manifest)
    summary = {"status": "audit_complete", "run_dir": str(directory),
               "B_rows": len(development.row_ids), "negative_increments_retained": development.audit["target"]["negative_rows"],
               "protocol": splits["protocol"],
               "partitions": [{"fold": value["fold"], "development": value["development"]["n"],
                               "validation": value["validation"]["n"]} for value in splits["outer_partitions"]],
               "A_outcomes_accessed": False, "training_performed": False}
    if check_inputs:
        summary["A_predictor_check"] = audit["test_inputs"]
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cli = vars(args).copy()
        command = cli.pop("command")
        config_path = cli.pop("config")
        check = cli.pop("check_runtime")
        check_inputs = cli.pop("check_test_inputs", False)
        validation_only = cli.pop("validation_only", False)
        evaluation_dir = cli.pop("evaluation_dir", None)
        report_dir = cli.pop("report_dir", None)
        no_plots = cli.pop("no_plots", False)
        figure_set = cli.pop("figure_set", "diagnostics")
        fig1_label_size = cli.pop("fig1_label_size", 24)
        fig1_tick_size = cli.pop("fig1_tick_size", 16)
        config = load_config(cli, config_path)
        if check and check_inputs:
            raise ValueError("--check-runtime cannot access data; omit --check-test-inputs")
        if config.resume and (command not in ("run", "train", "refit") or check):
            raise ValueError("--resume is allowed only for run/train/refit, not runtime checks")
        if command == "refit" and config.run_dir is None:
            raise ValueError("refit requires an explicit completed validation --run-dir")
        from .runtime import (bootstrap_threads, configure_torch, dependency_preflight,
                              runtime_manifest, save_runtime_manifest)
        limits = bootstrap_threads(config.threads)
        if command in ("report", "plots") and not check:
            if config.run_dir is None:
                raise ValueError("report/plots require a completed --run-dir")
            from pathlib import Path
            if evaluation_dir is not None and not Path(evaluation_dir).is_absolute():
                evaluation_dir = str(PROJECT_ROOT / evaluation_dir)
            if command == "report":
                from .reporting import write_report
                result = write_report(config.run_dir, evaluation_dir=evaluation_dir, report_dir=report_dir)
            else:
                dependency_preflight(["matplotlib", "seaborn"])
                from .plotting import plot_saved_results
                result = plot_saved_results(config.run_dir, evaluation_dir=evaluation_dir, report_dir=report_dir, figure_set=figure_set,
                                            fig1_label_size=fig1_label_size, fig1_tick_size=fig1_tick_size)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if command == "evaluate" and not check:
            from .evaluation import evaluate
            result = evaluate(config, evaluation_dir=evaluation_dir)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        requirements = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
        if command in ("run", "train", "refit"):
            requirements += ["tqdm"]
        if command in ("run", "train", "refit") and config.model_scope in ("all", "neural"):
            requirements += ["torch"]
        elif command == "plots":
            requirements += ["matplotlib", "seaborn"]
        versions = dependency_preflight(requirements)
        if command == "run" and not no_plots:
            versions.update(dependency_preflight(["matplotlib", "seaborn"]))
        classical_registry = None
        neural_settings = None
        if command in ("run", "train", "refit"):
            requirements += ["tqdm"]
        if command in ("run", "train", "refit") and config.model_scope in ("all", "classical"):
            from .models.classical import build_registry, load_classical_settings
            from .data import file_fingerprint
            settings = load_classical_settings(config.classical_config)
            registry = build_registry(settings, config.seed)
            classical_registry = {"enabled_models": settings["enabled_models"],
                                  "available_models": list(registry.estimators), "skipped": registry.skipped,
                                  "settings_fingerprint": file_fingerprint(config.classical_config)}
            from importlib import metadata
            for name, package in (("XGBoost", "xgboost"), ("LightGBM", "lightgbm")):
                if name in registry.estimators:
                    versions[package] = metadata.version(package)
        if command in ("run", "train", "refit"):
            requirements += ["tqdm"]
        if command in ("run", "train", "refit") and config.model_scope in ("all", "neural"):
            from .neural import load_neural_settings
            from .data import file_fingerprint
            neural_settings = load_neural_settings(config.neural_config)
        # Audit/plots need no CUDA, even though the training device remains recorded in config.
        needs_device = command in ("run", "train", "refit") and config.model_scope in ("all", "neural")
        device = configure_torch(config.device) if needs_device else {
            "requested": config.device, "selected": None, "reason": "No neural device needed for this command/model scope"}
        manifest = runtime_manifest(config, command, limits, versions, device, PROJECT_ROOT)
        if classical_registry is not None:
            manifest["classical_registry"] = classical_registry
        if neural_settings is not None:
            manifest["neural_settings"] = neural_settings
            manifest["neural_settings_fingerprint"] = file_fingerprint(config.neural_config)
        if command == "audit" and not check:
            return audit_command(config, manifest, check_inputs)
        if command == "run" and not check:
            from .workflow import run_pipeline
            result = run_pipeline(config, runtime=manifest, no_plots=no_plots)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["status"] == "pipeline_complete" else 2
        if command in ("train", "refit") and not check:
            from pathlib import Path
            from .training import final_refit, run_training
            refit_started = config.run_dir is not None and (Path(config.run_dir) / "refit_manifest.json").exists()
            if command == "refit" or (config.resume and refit_started and not validation_only):
                result = final_refit(config)
            else:
                result = run_training(config, runtime=manifest)
                if result["status"] == "validation_complete" and not validation_only:
                    result = final_refit(replace(config, run_dir=result["run_dir"]))
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["status"] in ("validation_complete", "frozen") else 2
        if config.run_dir:
            print(f"Runtime check only; manifest: {save_runtime_manifest(manifest, config.run_dir)}")
        else:
            print(json.dumps(manifest, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

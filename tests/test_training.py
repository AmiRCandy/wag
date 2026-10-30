"""B-only orchestration on tiny synthetic CSVs; no official datasets or A scores."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from wag.config import Config, PROJECT_ROOT
from wag.cpu import ModelResult
from wag.data import TARGET, file_fingerprint, load_development, make_split_manifest
from wag.models.classical import load_classical_settings
from wag.neural import NeuralResult, load_neural_settings
from wag.runtime import THREAD_ENV
from wag.training import (_execute_task, _models, _relative_folds, regression_metrics,
                          run_training, summarize_results)
from test_data import records, write_csv


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.b, self.a = self.root / "B.csv", self.root / "A_test.csv"
        rows = records(30)
        for i, row in enumerate(rows):
            row["hcpvi_gas_frac"] = .01 + i * .005
        write_csv(self.b, rows)
        self.settings = load_classical_settings()
        self.settings.update(enabled_models=["Ridge"], variants=["plain"])
        self.classical_path = self.root / "classical.json"
        self.classical_path.write_text(json.dumps(self.settings))
        self.config = Config(train_csv=str(self.b), test_csv=str(self.a), run_dir=str(self.root / "run"),
                             classical_config=str(self.classical_path), model_scope="classical", device="cpu", workers=2,
                             inner_folds=2, outer_folds=3, search_candidates=2)
        environment = patch.dict(os.environ, {key: "1" for key in THREAD_ENV})
        environment.start()
        self.addCleanup(environment.stop)

    def read(self, name):
        return json.loads((Path(self.config.run_dir) / name).read_text())

    def test_holdout_complete_without_A_and_recomputable_metrics(self):
        fingerprint = file_fingerprint(self.b)
        summary = run_training(self.config)
        self.assertEqual(summary["status"], "validation_complete")
        self.assertEqual(summary["primary_model_id"], "Ridge")
        self.assertFalse(summary["models_frozen"])
        self.assertFalse(self.a.exists())
        self.assertEqual(file_fingerprint(self.b), fingerprint)
        rows = pd.read_csv(Path(self.config.run_dir) / "validation_predictions.csv")
        self.assertEqual(len(rows), 6)
        np.testing.assert_allclose(rows.residual, rows.observed - rows.prediction)
        metrics = pd.read_csv(Path(self.config.run_dir) / "validation_metrics.csv").iloc[0]
        calculated = regression_metrics(rows.observed, rows.prediction)
        for key, value in calculated.items():
            self.assertAlmostEqual(metrics[key], value, places=10)
        task = self.read("task_records.json")[0]
        self.assertFalse(set(task["training_row_ids"]) & set(task["validation_row_ids"]))
        self.assertFalse((Path(self.config.run_dir) / "cv_fold_metrics.csv").exists())

    def test_A_content_changes_cannot_change_selection_or_B_predictions(self):
        self.a.write_text("unreadable A labels\n")
        first = run_training(self.config)
        selected = self.read("selection.json")
        prediction = (Path(first["run_dir"]) / "validation_predictions.csv").read_bytes()
        self.a.write_text("different A outcomes\n")
        second = run_training(replace(self.config, run_dir=str(self.root / "other")))
        self.assertEqual(json.loads((Path(second["run_dir"]) / "selection.json").read_text()), selected)
        self.assertEqual((Path(second["run_dir"]) / "validation_predictions.csv").read_bytes(), prediction)

    def test_nested_CV_exactly_once_OOF_fold_and_pooled_metrics(self):
        self.config = replace(self.config, validation="nested-cv")
        run_training(self.config)
        rows = pd.read_csv(Path(self.config.run_dir) / "cv_oof_predictions.csv")
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows.row_id.nunique(), 30)
        self.assertEqual(set(rows.position), set(range(30)))
        source = pd.read_csv(self.b)
        np.testing.assert_allclose(rows.observed, source[TARGET].iloc[rows.position])
        folds = pd.read_csv(Path(self.config.run_dir) / "cv_fold_metrics.csv")
        aggregate = pd.read_csv(Path(self.config.run_dir) / "validation_metrics.csv").iloc[0]
        self.assertAlmostEqual(aggregate.mean_outer_RMSE, folds.RMSE.mean())
        self.assertAlmostEqual(aggregate.std_outer_RMSE, folds.RMSE.std(ddof=0))
        self.assertAlmostEqual(aggregate.RMSE, regression_metrics(rows.observed, rows.prediction)["RMSE"])
        self.assertAlmostEqual(self.read("selection.json")["ranking"][0]["selection_RMSE"], aggregate.mean_outer_RMSE, places=12)

    def test_supplied_inner_folds_are_relative_and_inside_development(self):
        data = load_development(self.b, test_path=self.a)
        partition = make_split_manifest(data, self.config)["outer_partitions"][0]
        positions = partition["development"]["positions"]
        relative = _relative_folds(partition)
        for pair, original in zip(relative, partition["inner_folds"]):
            self.assertEqual([positions[i] for i in pair[0]], original["train"]["positions"])
            self.assertEqual([positions[i] for i in pair[1]], original["validation"]["positions"])

    def test_real_tiny_search_uses_saved_inner_CV_and_stable_trial_IDs(self):
        settings = deepcopy(self.settings)
        settings["enabled_models"] = ["RandomForest"]
        settings["models"]["RandomForest"]["search"] = {"n_estimators": [2, 3], "max_depth": [2]}
        self.classical_path.write_text(json.dumps(settings))
        run_training(self.config)
        tuning = self.read("task_records.json")[0]["tuning"]
        self.assertEqual(tuning["cv_source"], "supplied_partition_indices")
        self.assertEqual(tuning["score_role"], "inner_tuning_only")
        self.assertEqual(len({r["trial_id"] for r in tuning["trials"]}), 2)
        self.assertEqual(tuning["workers"], 2)
        self.assertEqual(tuning["native_threads"], 1)

    def test_completed_resume_performs_no_fits(self):
        run_training(self.config)
        before = (Path(self.config.run_dir) / "validation_predictions.csv").read_bytes()
        with patch("wag.training._execute_task", side_effect=AssertionError("Should reuse completed task")):
            summary = run_training(replace(self.config, resume=True))
        self.assertEqual(summary["status"], "validation_complete")
        self.assertEqual(before, (Path(self.config.run_dir) / "validation_predictions.csv").read_bytes())

    def test_interruption_resume_reuses_success_and_fits_only_unfinished_task(self):
        self.settings["enabled_models"] = ["Ridge", "LinearRegression"]
        self.classical_path.write_text(json.dumps(self.settings))
        calls = []
        def interrupt(*args):
            calls.append(args[-1][0])
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return _execute_task(*args)
        with patch("wag.training._execute_task", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            run_training(self.config)
        self.assertEqual(self.read("training_manifest.json")["status"], "running")
        self.assertFalse((Path(self.config.run_dir) / ".training.lock").exists())
        with patch("wag.training._execute_task", wraps=_execute_task) as fitter:
            run_training(replace(self.config, resume=True))
        self.assertEqual(fitter.call_count, 1)
        self.assertEqual(fitter.call_args.args[-1][0], "LinearRegression")
        tasks = self.read("task_records.json")
        self.assertEqual([r["attempts"] for r in tasks], [1, 2])

    def test_failed_task_retry_is_bounded_to_two_attempts(self):
        with patch("wag.training.search_models", side_effect=RuntimeError("fit failed")) as fitter:
            self.assertEqual(run_training(self.config)["status"], "validation_incomplete")
            run_training(replace(self.config, resume=True))
            run_training(replace(self.config, resume=True))
        self.assertEqual(fitter.call_count, 2)
        self.assertIsNone(self.read("selection.json")["primary_model_id"])
        self.assertFalse(self.read("task_records.json")[0]["fallback_performed"])

    def test_resume_refuses_data_settings_code_and_configuration_changes(self):
        run_training(self.config)
        with self.assertRaisesRegex(ValueError, "Resume requires"):
            run_training(replace(self.config, resume=True, seed=6))
        with patch("wag.training.file_fingerprint", side_effect=lambda p: "different" if Path(p).name == "training.py" else file_fingerprint(p)):
            with self.assertRaisesRegex(ValueError, "Resume requires"):
                run_training(replace(self.config, resume=True))
        original = self.classical_path.read_text()
        settings = deepcopy(self.settings)
        settings["models"]["Ridge"]["defaults"]["alpha"] = 2
        self.classical_path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "Resume requires"):
            run_training(replace(self.config, resume=True))
        self.classical_path.write_text(original)
        rows = records(30)
        rows[0]["rf_incremental_pct"] += 1
        rows[0]["rf_wag_total_pct"] += 1
        write_csv(self.b, rows)
        with self.assertRaisesRegex(ValueError, "Resume requires"):
            run_training(replace(self.config, resume=True))

    def test_corrupt_task_and_split_artifacts_refused(self):
        run_training(self.config)
        task_path = next((Path(self.config.run_dir) / "tasks").glob("*.json"))
        if task_path.name.endswith(".attempt.json"):
            task_path = next(p for p in task_path.parent.glob("*.json") if not p.name.endswith(".attempt.json"))
        original = task_path.read_text()
        payload = json.loads(original)
        payload["result"]["predictions"][0]["prediction"] += 10
        task_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "checksum"):
            run_training(replace(self.config, resume=True))
        task_path.write_text(original)
        split = Path(self.config.run_dir) / "split_manifest.json"
        split.write_text(split.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "split artifact changed"):
            run_training(replace(self.config, resume=True))

    def test_no_overwrite_existing_run_and_stale_lock_blocks_resume(self):
        run_training(self.config)
        with self.assertRaises(FileExistsError):
            run_training(self.config)
        lock = Path(self.config.run_dir) / ".training.lock"
        lock.write_text("another process")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            run_training(replace(self.config, resume=True))
        self.assertEqual(lock.read_text(), "another process")

    def test_malformed_resume_objects_fail_with_clear_schema_errors(self):
        run_training(self.config)
        manifest = Path(self.config.run_dir) / "training_manifest.json"
        original = manifest.read_text()
        manifest.write_text("[]")
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            run_training(replace(self.config, resume=True))
        manifest.write_text(original)
        task = next(p for p in (Path(self.config.run_dir) / "tasks").glob("*.json") if not p.name.endswith(".attempt.json"))
        task.write_text("[]")
        with self.assertRaisesRegex(ValueError, "object schema"):
            run_training(replace(self.config, resume=True))

    def test_one_failed_outer_fold_excludes_model_from_nested_ranking(self):
        self.settings["enabled_models"] = ["Ridge", "LinearRegression"]
        self.classical_path.write_text(json.dumps(self.settings))
        def fail(data, partition, config, classical, neural, model):
            if partition["fold"] == 1 and model[0] == "Ridge":
                return {"model_id": "Ridge", "fold": 1, "status": "failed", "predictions": [], "metrics": None}
            return _execute_task(data, partition, config, classical, neural, model)
        with patch("wag.training._execute_task", side_effect=fail):
            summary = run_training(replace(self.config, validation="nested-cv"))
        self.assertEqual(summary["status"], "validation_incomplete")
        selection = self.read("selection.json")
        self.assertEqual(selection["primary_model_id"], "LinearRegression")
        self.assertEqual(selection["excluded_models"][0]["model_id"], "Ridge")

    def test_metric_constant_target_R2_undefined_and_invalid_predictions(self):
        self.assertIsNone(regression_metrics([1, 1], [1, 1])["R2"])
        self.assertEqual(regression_metrics([0, 1], [2, 3])["bias"], 2)
        for y, p in (([], []), ([1, 2], [1]), ([1], [np.nan])):
            with self.assertRaises(ValueError):
                regression_metrics(y, p)

    def test_fixed_selection_ties_use_MAE_then_model_ID_not_R2(self):
        def result(name, metric):
            return {"model_id": name, "fold": 0, "status": "validated", "metrics": metric,
                    "predictions": [{"row_id": "one", "observed": 0., "prediction": 1.}, {"row_id": "two", "observed": 2., "prediction": 1.}]}
        a = result("z", regression_metrics([0, 2], [1, 1]))
        b = result("a", regression_metrics([0, 2], [1, 1]))
        selection = summarize_results([a, b], "holdout", ["z", "a"], 1, ["one", "two"])[-1]
        self.assertEqual(selection["primary_model_id"], "a")
        a["predictions"] = [{"row_id": "one", "observed": 0., "prediction": np.sqrt(2.)},
                            {"row_id": "two", "observed": 2., "prediction": 2.}]
        a["metrics"] = regression_metrics([0, 2], [np.sqrt(2.), 2.])
        selection = summarize_results([a, b], "holdout", ["z", "a"], 1, ["one", "two"])[-1]
        self.assertEqual(selection["primary_model_id"], "z")

    def test_nested_ranking_uses_mean_not_pooled_RMSE(self):
        results = []
        for model, errors in (("A", [0., 2.]), ("B", [1.1, 1.1])):
            for fold, error in enumerate(errors):
                results.append({"model_id": model, "fold": fold, "status": "validated", "metrics": regression_metrics([0.], [error]),
                                "predictions": [{"row_id": str(fold), "observed": 0., "prediction": error}]})
        aggregates, selection = summarize_results(results, "nested-cv", ["A", "B"], 2, ["0", "1"])[2:]
        self.assertGreater(aggregates[0]["RMSE"], aggregates[1]["RMSE"])
        self.assertEqual(selection["primary_model_id"], "A")

    def test_neural_inner_selection_then_development_refit_excludes_outer(self):
        data = load_development(self.b, test_path=self.a)
        partition = make_split_manifest(data, self.config)["outer_partitions"][0]
        settings = load_neural_settings()
        calls = []
        class Predictor:
            def predict(self, X):
                return np.full(len(X), -1.)
        def neural(X, y, config, settings, name, **kwargs):
            calls.append((X.index.tolist(), y.index.tolist(), kwargs))
            return NeuralResult(name, "fitted", Predictor(), {"selected_epochs": 7, "history": []})
        with patch("wag.training.fit_neural", side_effect=neural):
            result = _execute_task(data, partition, self.config, None, settings, ("NN_MLP", "neural", None, None))
        self.assertEqual(result["status"], "validated")
        self.assertEqual(len(calls), 2)
        outer = set(partition["validation"]["positions"])
        self.assertFalse(set(calls[0][0]) & outer)
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertEqual(calls[1][2]["epochs"], 7)
        self.assertEqual(calls[1][2]["phase"], "development_refit")
        self.assertFalse(set(result["inner_training_row_ids"]) & set(result["inner_validation_row_ids"]))

    def test_neural_failure_does_not_refit_or_fall_back(self):
        data = load_development(self.b, test_path=self.a)
        partition = make_split_manifest(data, self.config)["outer_partitions"][0]
        with patch("wag.training.fit_neural", return_value=NeuralResult("NN_MLP", "failed", details={"error": "OOM"})) as fitter:
            result = _execute_task(data, partition, self.config, None, load_neural_settings(), ("NN_MLP", "neural", None, None))
        fitter.assert_called_once()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["fallback_performed"])

    def test_classical_only_CLI_no_PyTorch_no_A(self):
        output = self.root / "cli"
        command = [sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "train", "--model-scope", "classical",
                   "--classical-config", str(self.classical_path), "--train-csv", str(self.b), "--test-csv", str(self.a),
                   "--run-dir", str(output), "--workers", "2", "--inner-folds", "2"]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "training_manifest.json").read_text())
        self.assertIsNone(manifest["device"]["selected"])
        self.assertNotIn("torch", manifest["packages"])
        self.assertFalse(manifest["A_outcomes_accessed"])
        again = subprocess.run(command + ["--resume"], cwd=self.root, capture_output=True, text=True, timeout=45)
        self.assertEqual(again.returncode, 0, again.stderr)

    def test_training_module_import_does_not_open_data_or_import_torch(self):
        code = '''
from wag.runtime import bootstrap_threads
bootstrap_threads()
import sys, tempfile, os, pathlib, multiprocessing
with tempfile.TemporaryDirectory() as directory:
    os.chdir(directory)
    import wag.training
    assert 'torch' not in sys.modules
    assert not multiprocessing.active_children()
    assert list(pathlib.Path(directory).iterdir()) == []
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_neural_tiny_end_to_end_when_PyTorch_available(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch unavailable; real neural orchestration not verified")
        settings = load_neural_settings()
        settings.update(hidden_layers=[4], max_epochs=2)
        settings["constraints"]["collocation_count"] = 4
        settings["early_stopping"]["patience"] = 1
        path = self.root / "neural.json"
        path.write_text(json.dumps(settings))
        summary = run_training(replace(self.config, model_scope="neural", neural_config=str(path)))
        self.assertEqual(summary["status"], "validation_complete")
        tasks = self.read("task_records.json")
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            self.assertLessEqual(task["selected_epochs"], 2)
            self.assertEqual(task["development_refit"]["selected_epochs"], task["selected_epochs"])
            self.assertEqual(task["development_refit"]["preprocessing"]["fit_rows"], 24)
        history = pd.read_csv(Path(self.config.run_dir) / "training_history.csv")
        self.assertEqual(set(history.stage), {"tuning", "development_refit"})


if __name__ == "__main__":
    unittest.main()

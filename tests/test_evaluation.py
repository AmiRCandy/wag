"""Synthetic external evaluation only; official A outcomes are never opened."""

from copy import deepcopy
from dataclasses import replace
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from wag.artifacts import load_frozen_models
from wag.config import Config, PROJECT_ROOT
from wag.data import file_fingerprint
from wag.evaluation import evaluate, metrics
from wag.models.classical import load_classical_settings
from wag.runtime import THREAD_ENV
from wag.training import final_refit, run_training
from test_data import records, write_csv


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.b, self.a = self.root / "B.csv", self.root / "A.csv"
        B = records(30)
        for i, row in enumerate(B):
            row["hcpvi_gas_frac"] = .01 + i * .005
        write_csv(self.b, B)
        self.A = records(12, family="A")
        for i, row in enumerate(self.A):
            row["hcpvi_gas_frac"] = .01 + i * .005
        self.A[-1].update(perm_mult=2, effective_perm_md=500)
        write_csv(self.a, self.A)
        settings = load_classical_settings()
        settings.update(enabled_models=["Ridge"], variants=["plain", "guided"])
        self.settings = self.root / "classical.json"
        self.settings.write_text(json.dumps(settings))
        self.run = self.root / "run"
        self.config = Config(train_csv=str(self.b), test_csv=str(self.a), run_dir=str(self.run),
                             classical_config=str(self.settings), model_scope="classical", device="cpu", workers=1,
                             inner_folds=2, search_candidates=2)
        environment = patch.dict(os.environ, {name: "1" for name in THREAD_ENV})
        environment.start()
        self.addCleanup(environment.stop)

    def freeze(self):
        self.assertEqual(run_training(self.config)["status"], "validation_complete")
        final_refit(self.config)
        return load_frozen_models(self.run)

    def outputs(self, result):
        directory = Path(result["evaluation_dir"])
        return (pd.read_csv(directory / "test_predictions.csv"), pd.read_csv(directory / "test_metrics.csv"),
                json.loads((directory / "evaluation_manifest.json").read_text()))

    def test_every_row_metrics_raw_outputs_primary_and_coverage(self):
        frozen = self.freeze()
        before = {path: file_fingerprint(self.run / path) for path in frozen.manifest["artifact_fingerprints"]}
        predictions, scores, manifest = self.outputs(evaluate(self.config))
        self.assertEqual(len(predictions), 24)
        self.assertEqual(set(predictions.model_id), {"Ridge", "Ridge_GUIDED"})
        self.assertEqual(predictions.groupby("model_id").size().tolist(), [12, 12])
        self.assertEqual(manifest["primary_model_id"], frozen.manifest["primary_model_id"])
        self.assertFalse(manifest["training_performed"])
        self.assertEqual(manifest["status"], "evaluation_complete")
        self.assertTrue(predictions.ood_any.any())
        self.assertTrue(predictions.loc[predictions.position == 11, "ood_raw_range"].all())
        self.assertEqual(int((predictions.observed < 0).sum()), 20)
        for _, score in scores.iterrows():
            rows = predictions[predictions.model_id == score.model_id]
            values = rows.prediction if score.output == "primary" else rows.raw_prediction
            computed = metrics(rows.observed, values)
            for name in ("RMSE", "MAE", "R2", "bias"):
                self.assertAlmostEqual(score[name], computed[name], places=10)
            np.testing.assert_allclose(rows.residual, rows.observed - rows.prediction)
        for path, fingerprint in before.items():
            self.assertEqual(file_fingerprint(self.run / path), fingerprint)
        for path, fingerprint in manifest["artifact_fingerprints"].items():
            self.assertEqual(file_fingerprint(Path(manifest["evaluation_dir"]) / path), fingerprint)
        self.assertIn("simulated", manifest["interpretation"])

    def test_no_fit_and_no_B_required_saved_scope_and_mode_control_inference(self):
        self.freeze()
        self.b.unlink()
        self.settings.unlink()
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No fitting allowed")), patch("wag.training.run_training", side_effect=AssertionError("No training")):
            result = evaluate(replace(self.config, model_scope="neural", injection_mode="rates", device="cuda:0"))
        self.assertEqual(result["A_rows"], 12)
        self.assertEqual(set(result["model_ids"]), {"Ridge", "Ridge_GUIDED"})

    def test_permuting_A_labels_changes_only_scores_and_residuals(self):
        self.freeze()
        first, first_scores, _ = self.outputs(evaluate(self.config))
        rows = deepcopy(self.A)
        labels = [r["rf_incremental_pct"] for r in rows][::-1]
        for row, label in zip(rows, labels):
            row.update(rf_incremental_pct=label, rf_wag_total_pct=40 + label)
        write_csv(self.a, rows)
        second, second_scores, _ = self.outputs(evaluate(self.config))
        for name in ("row_id", "prediction", "raw_prediction", "ood_raw_range", "ood_scaled"):
            pd.testing.assert_series_equal(first[name], second[name])
        self.assertFalse(np.allclose(first_scores.RMSE, second_scores.RMSE))
        self.assertFalse(np.allclose(first.residual, second.residual))

    def test_inference_receives_no_outcomes_metadata_or_extra_columns(self):
        self.freeze()
        rows = deepcopy(self.A)
        for row in rows:
            row["future_outcome"] = 999
        pd.DataFrame(rows).to_csv(self.a, index=False)
        original = Pipeline.predict
        calls = []
        def predict(model, X, *args, **kwargs):
            calls.append(len(X))
            for column in ("rf_incremental_pct", "rf_wag_total_pct", "rf_waterflood_total_pct",
                           "cum_oil_stb", "oip_initial_stb", "source", "reservoir_family", "future_outcome"):
                self.assertNotIn(column, X.columns)
            return original(model, X, *args, **kwargs)
        with patch.object(Pipeline, "predict", predict):
            evaluate(self.config)
        self.assertIn(12, calls)

    def test_frozen_marker_is_verified_before_external_read(self):
        self.run.mkdir()
        with patch("wag.evaluation.load_external", side_effect=AssertionError("Must not read A")) as loader:
            with self.assertRaises(ValueError):
                evaluate(self.config)
        loader.assert_not_called()
        self.assertFalse((self.run / "evaluations").exists())

    def test_invalid_dataset_rejected_as_a_whole_and_failure_recorded(self):
        self.freeze()
        for label, mutate in (("wrong-family", lambda r: r.update(reservoir_family="B")),
                              ("bad-target", lambda r: r.update(rf_incremental_pct=999)),
                              ("bad-context", lambda r: r.update(grid_nx=2)),
                              ("bad-value", lambda r: r.update(effective_perm_md="bad"))):
            rows = deepcopy(self.A)
            mutate(rows[0])
            write_csv(self.a, rows)
            directory = self.root / label
            with self.assertRaises(ValueError):
                evaluate(self.config, evaluation_dir=directory)
            manifest = json.loads((directory / "evaluation_manifest.json").read_text())
            self.assertEqual(manifest["status"], "evaluation_failed")
            self.assertTrue(manifest["rejected_rows"])
            self.assertFalse((directory / "test_metrics.csv").exists())

    def test_duplicate_or_missing_columns_do_not_get_cleaned(self):
        self.freeze()
        write_csv(self.a, [self.A[0], self.A[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate(self.config)
        write_csv(self.a, self.A)
        frame = pd.read_csv(self.a).drop(columns="rf_incremental_pct")
        frame.to_csv(self.a, index=False)
        with self.assertRaisesRegex(ValueError, "Missing required"):
            evaluate(self.config)

    def test_rerun_is_new_event_and_explicit_directory_never_overwritten(self):
        self.freeze()
        first, second = evaluate(self.config), evaluate(self.config)
        self.assertNotEqual(first["evaluation_dir"], second["evaluation_dir"])
        with self.assertRaises(FileExistsError):
            evaluate(self.config, evaluation_dir=first["evaluation_dir"])
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            evaluate(replace(self.config, resume=True))

    def test_csv_mutation_during_scoring_never_completes(self):
        self.freeze()
        from wag.evaluation import coverage
        def mutate(*args, **kwargs):
            rows = deepcopy(self.A)
            rows[0].update(rf_incremental_pct=1, rf_wag_total_pct=41)
            write_csv(self.a, rows)
            return coverage(*args, **kwargs)
        directory = self.root / "mutated"
        with patch("wag.evaluation.coverage", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                evaluate(self.config, evaluation_dir=directory)
        self.assertEqual(json.loads((directory / "evaluation_manifest.json").read_text())["status"], "evaluation_failed")

    def test_empty_subgroups_and_constant_target_metrics_are_explicit(self):
        self.assertEqual(metrics([], [])["n"], 0)
        self.assertIsNone(metrics([1, 1], [2, 3])["R2"])
        self.assertIsNone(metrics([1], [2])["R2"])
        with self.assertRaises(ValueError):
            metrics([1], [float("nan")])

    def test_classical_CLI_evaluates_without_PyTorch_or_training_data(self):
        self.freeze()
        self.b.unlink()
        process = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "evaluate", "--run-dir", str(self.run),
                                  "--test-csv", str(self.a), "--evaluation-dir", str(self.root / "cli")],
                                 cwd=self.root, capture_output=True, text=True, env=os.environ.copy())
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["A_rows"], 12)

    def test_evaluation_import_has_no_IO_torch_or_pool(self):
        program = "import builtins, pathlib, sys; import sklearn.metrics, pandas, wag.artifacts; from unittest.mock import patch; " + \
                  "guard=patch.object(pathlib.Path,'open',side_effect=AssertionError('No IO')); guard.start(); " + \
                  "import wag.evaluation; assert 'torch' not in sys.modules; " + \
                  "import multiprocessing; assert not multiprocessing.active_children()"
        process = subprocess.run([sys.executable, "-B", "-c", program], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_real_neural_evaluation_when_PyTorch_is_available(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch unavailable; real neural external inference not verified")
        from wag.neural import load_neural_settings
        settings = load_neural_settings()
        settings.update(hidden_layers=[4], max_epochs=2)
        settings["constraints"]["collocation_count"] = 4
        path = self.root / "neural.json"
        path.write_text(json.dumps(settings))
        self.config = replace(self.config, model_scope="neural", neural_config=str(path))
        self.freeze()
        predictions, scores, state = self.outputs(evaluate(self.config))
        self.assertEqual(set(predictions.model_id), {"NN_MLP", "NN_PINN_HARD"})
        self.assertEqual(len(predictions), 24)
        self.assertTrue(np.isfinite(predictions.prediction).all())
        self.assertFalse(state["training_performed"])


if __name__ == "__main__":
    unittest.main()

"""Synthetic all-B refits, atomic freeze, portable loading and rejection paths."""

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

from wag.artifacts import (FROZEN_NAME, canonical_hash, load_frozen_models, publish_frozen_manifest,
                           safe_artifact_path)
from wag.config import Config, PROJECT_ROOT
from wag.cpu import ModelResult, _fit_one
from wag.data import file_fingerprint
from wag.models.classical import load_classical_settings
from wag.neural import load_neural_settings
from wag.runtime import THREAD_ENV
from wag.training import final_refit, run_training
from test_data import records, write_csv


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.b, self.a = self.root / "B.csv", self.root / "A_test.csv"
        rows = records(30)
        for i, row in enumerate(rows):
            row["hcpvi_gas_frac"] = .01 + i * .005
        write_csv(self.b, rows)
        self.settings = load_classical_settings()
        self.settings.update(enabled_models=["Ridge"], variants=["plain", "guided"])
        self.path = self.root / "classical.json"
        self.path.write_text(json.dumps(self.settings))
        self.directory = self.root / "run"
        self.config = Config(train_csv=str(self.b), test_csv=str(self.a), run_dir=str(self.directory),
                             classical_config=str(self.path), model_scope="classical", device="cpu", workers=1,
                             inner_folds=2, outer_folds=3, search_candidates=2)
        environment = patch.dict(os.environ, {key: "1" for key in THREAD_ENV})
        environment.start()
        self.addCleanup(environment.stop)

    def validate(self):
        self.assertEqual(run_training(self.config)["status"], "validation_complete")

    def freeze(self):
        self.validate()
        result = final_refit(self.config)
        self.assertEqual(result["status"], "frozen")
        return load_frozen_models(self.directory)

    def read(self, name):
        return json.loads((self.directory / name).read_text())

    def test_all_B_refit_metadata_winner_negative_targets_and_portable_loading(self):
        before = file_fingerprint(self.b)
        frozen = self.freeze()
        self.assertEqual(frozen.manifest["B"]["rows"], 30)
        self.assertEqual(frozen.manifest["B"]["fingerprint"], before)
        self.assertEqual(set(frozen.models), {"Ridge", "Ridge_GUIDED"})
        self.assertEqual(frozen.manifest["primary_model_id"], self.read("selection.json")["primary_model_id"])
        self.assertFalse(frozen.manifest["A_outcomes_accessed"])
        for information in frozen.metadata.values():
            self.assertEqual(information["preprocessing"]["fit_rows"], 30)
            self.assertIn("median_imputation", information["preprocessing"])
            self.assertTrue(information["configuration_hash"])
            self.assertEqual(information["inference"]["OOD"]["action"], "diagnose_only")
        self.assertEqual(frozen.metadata["Ridge_GUIDED"]["inference"]["bounds"], [-100., 100.])
        self.assertEqual(self.read("data_audit.json")["target"]["negative_rows"], 10)
        self.assertEqual(file_fingerprint(self.b), before)
        self.assertFalse(self.a.exists())
        self.assertFalse(self.read("training_manifest.json")["models_frozen"])

    def test_holdout_never_retunes_during_final_refit(self):
        self.validate()
        with patch("wag.training.search_models", side_effect=AssertionError("Holdout parameters must be reused")):
            final_refit(self.config)
        plans = self.read("refit_plan.json")["plans"]
        self.assertTrue(all(p["source"] == "holdout_development_selection" for p in plans.values()))

    def test_nested_refit_uses_all_B_inner_tuning_and_preserves_original_winner(self):
        self.config = replace(self.config, validation="nested-cv")
        self.validate()
        selected = self.read("selection.json")["primary_model_id"]
        from wag.training import search_models
        counts = []
        def search(X, y, config, settings, **kwargs):
            counts.append((len(X), sum(len(valid) for _, valid in kwargs["cv"])))
            return search_models(X, y, config, settings, **kwargs)
        with patch("wag.training.search_models", side_effect=search):
            final_refit(self.config)
        self.assertEqual(counts, [(30, 30), (30, 30)])
        frozen = load_frozen_models(self.directory)
        self.assertEqual(frozen.manifest["primary_model_id"], selected)
        self.assertTrue(all(p["source"] == "all_B_inner_tuning" for p in self.read("refit_plan.json")["plans"].values()))

    def test_A_changes_cannot_change_final_model_predictions_or_parameters(self):
        self.a.write_text("unreadable A labels")
        frozen = self.freeze()
        probe = self.read("refit_probe.json")
        X = pd.DataFrame(probe["inputs"])
        predictions = {name: model.predict(X) for name, model in frozen.models.items()}
        self.a.write_text("different A outcomes")
        other = self.root / "second"
        config = replace(self.config, run_dir=str(other))
        run_training(config)
        final_refit(config)
        second = load_frozen_models(other)
        for name in predictions:
            np.testing.assert_array_equal(second.models[name].predict(X), predictions[name])
            self.assertEqual(second.metadata[name]["configuration"], frozen.metadata[name]["configuration"])

    def test_loading_needs_no_B_or_A_and_predicts_in_a_fresh_CPU_interpreter(self):
        frozen = self.freeze()
        self.b.unlink()
        probe = self.read("refit_probe.json")
        X = pd.DataFrame(probe["inputs"])
        expected = frozen.models["Ridge"].predict(X)
        actual = load_frozen_models(self.directory).models["Ridge"].predict(X)
        np.testing.assert_array_equal(actual, expected)
        code = '''
from wag.runtime import bootstrap_threads
bootstrap_threads()
import sys, json, pathlib, pandas as pd, numpy as np
from wag.artifacts import load_frozen_models
directory = pathlib.Path(sys.argv[1])
frozen = load_frozen_models(directory, device='cpu')
probe = json.loads((directory / 'refit_probe.json').read_text())
for name, model in frozen.models.items():
    np.testing.assert_allclose(model.predict(pd.DataFrame(probe['inputs'])), frozen.manifest['models'][name]['verification']['prediction'], rtol=1e-5, atol=1e-4)
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.directory)], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_repeat_is_read_only_and_validation_cannot_resume(self):
        self.freeze()
        original = (self.directory / FROZEN_NAME).read_bytes()
        with patch("wag.cpu._fit_one", side_effect=AssertionError("Frozen models must not refit")):
            self.assertEqual(final_refit(replace(self.config, resume=True))["status"], "frozen")
        self.assertEqual(original, (self.directory / FROZEN_NAME).read_bytes())
        with self.assertRaisesRegex(ValueError, "Validation cannot resume"):
            run_training(replace(self.config, resume=True))

    def test_missing_or_corrupted_bundle_never_loads(self):
        frozen = self.freeze()
        path = self.directory / frozen.manifest["models"]["Ridge"]["files"]["model"]
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            load_frozen_models(self.directory)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            load_frozen_models(self.directory)

    def test_manifest_corruption_and_changed_winner_are_rejected(self):
        frozen = self.freeze()
        manifest = deepcopy(frozen.manifest)
        manifest["primary_model_id"] = "other"
        (self.directory / FROZEN_NAME).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_frozen_models(self.directory)
        names = frozen.manifest["model_ids"]
        manifest["primary_model_id"] = next(n for n in names if n != frozen.manifest["primary_model_id"])
        manifest["manifest_checksum"] = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_checksum"})
        (self.directory / FROZEN_NAME).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            load_frozen_models(self.directory)

    def test_incomplete_validation_or_changed_B_prevents_refit(self):
        self.validate()
        manifest = self.read("training_manifest.json")
        manifest["status"] = "validation_incomplete"
        (self.directory / "training_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "complete"):
            final_refit(self.config)
        manifest["status"] = "validation_complete"
        (self.directory / "training_manifest.json").write_text(json.dumps(manifest))
        rows = records(30)
        write_csv(self.b, rows)
        with self.assertRaisesRegex(ValueError, "differ from validation"):
            final_refit(self.config)
        self.assertFalse((self.directory / FROZEN_NAME).exists())

    def test_final_failure_never_freezes_and_resume_reuses_successful_bundle(self):
        self.validate()
        def fail_guided(family, variant, *args, **kwargs):
            if variant == "guided":
                return ModelResult(family + "_GUIDED", "failed", details={"error": "injected failure"})
            return _fit_one(family, variant, *args, **kwargs)
        with patch("wag.cpu._fit_one", side_effect=fail_guided):
            self.assertEqual(final_refit(self.config)["status"], "refit_incomplete")
        self.assertFalse((self.directory / FROZEN_NAME).exists())
        with self.assertRaisesRegex(ValueError, "explicit --resume"):
            final_refit(self.config)
        calls = []
        def fit(family, variant, *args, **kwargs):
            calls.append(variant)
            return _fit_one(family, variant, *args, **kwargs)
        with patch("wag.cpu._fit_one", side_effect=fit):
            self.assertEqual(final_refit(replace(self.config, resume=True))["status"], "frozen")
        self.assertEqual(calls, ["guided"])

    def test_serialization_failure_blocks_freeze_and_retry_cap_is_preserved(self):
        self.validate()
        with patch("wag.artifacts.save_model_bundle", side_effect=ValueError("round trip mismatch")) as save:
            self.assertEqual(final_refit(self.config)["status"], "refit_incomplete")
            self.assertEqual(final_refit(replace(self.config, resume=True))["status"], "refit_incomplete")
            self.assertEqual(final_refit(replace(self.config, resume=True))["status"], "refit_incomplete")
        self.assertEqual(save.call_count, 4)
        self.assertFalse((self.directory / FROZEN_NAME).exists())

    def test_safe_paths_and_incomplete_markers_are_rejected(self):
        self.directory.mkdir()
        for relative in ("../B.csv", str(self.b), "absent"):
            with self.assertRaises(ValueError):
                safe_artifact_path(self.directory, relative)
        alias = self.directory / "alias"
        alias.symlink_to(self.b)
        with self.assertRaises(ValueError):
            safe_artifact_path(self.directory, "alias")
        with self.assertRaises(ValueError):
            load_frozen_models(self.directory)

    def test_invalid_candidate_never_publishes_a_freeze_marker(self):
        self.directory.mkdir()
        with self.assertRaises(ValueError):
            publish_frozen_manifest(self.directory, {"status": "frozen", "models_frozen": True, "schema_version": 1})
        self.assertFalse((self.directory / FROZEN_NAME).exists())

    def test_classical_CLI_train_freezes_and_validation_only_stops(self):
        common = [sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "train", "--model-scope", "classical", "--device", "cpu",
                  "--classical-config", str(self.path), "--train-csv", str(self.b), "--test-csv", str(self.a), "--workers", "1", "--inner-folds", "2"]
        result = subprocess.run(common + ["--run-dir", str(self.directory)], cwd=self.root, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.directory / FROZEN_NAME).exists())
        second = self.root / "validation_only"
        result = subprocess.run(common + ["--validation-only", "--run-dir", str(second)], cwd=self.root, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((second / FROZEN_NAME).exists())

    def test_real_neural_CPU_bundle_when_PyTorch_is_available(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch unavailable; neural final refit/serialization not verified")
        settings = load_neural_settings()
        settings.update(hidden_layers=[4], max_epochs=2)
        settings["constraints"]["collocation_count"] = 4
        path = self.root / "neural.json"
        path.write_text(json.dumps(settings))
        self.config = replace(self.config, model_scope="neural", neural_config=str(path))
        frozen = self.freeze()
        for information in frozen.metadata.values():
            self.assertEqual(information["fit_record"]["epoch_policy"], "fixed_B_selected_budget")
            self.assertEqual(information["fit_record"]["preprocessing"]["fit_rows"], 30)
            self.assertLessEqual(information["configuration"]["selected_epochs"], 2)


if __name__ == "__main__":
    unittest.main()

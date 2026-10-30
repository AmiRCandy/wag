"""Synthetic whole-workflow integration and resumable stage boundaries."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from wag.data import file_fingerprint
from wag.workflow import run_pipeline
import test_reporting as reporting_tests


class WorkflowTests(reporting_tests.ReportingTests):
    # Use the reporting fixture, not its tests, to keep integration checks focused.
    test_holdout_report_truthful_exports_no_fitting_or_A_reads = None
    test_nested_report_separates_pooled_and_mean_fold_metrics = None
    test_test_event_linkage_and_saved_primary_stay_fixed = None
    test_corruption_failed_validation_and_existing_directory_rejected = None
    test_importance_recomputes_dependencies_and_is_seeded_B_only = None
    test_saved_result_rendering_uses_existing_style_and_no_refits = None
    test_original_font_preservation_when_available = None
    test_report_CLI_needs_no_GPU_data_or_PyTorch = None

    def test_full_no_plots_and_completed_resume_are_fitting_free(self):
        result = run_pipeline(self.config, no_plots=True)
        self.assertEqual(result["status"], "pipeline_complete")
        pipeline = self.run / "pipeline_manifest.json"
        before = file_fingerprint(pipeline)
        self.b.unlink()
        self.a.unlink()
        with patch("wag.training.run_training", side_effect=AssertionError("No development")), patch("wag.training.final_refit", side_effect=AssertionError("No refit")), patch("wag.evaluation.evaluate", side_effect=AssertionError("No A scoring")), patch("wag.reporting.write_report", side_effect=AssertionError("No report rerun")):
            again = run_pipeline(replace(self.config, resume=True), no_plots=True)
        self.assertEqual(again, result)
        self.assertEqual(file_fingerprint(pipeline), before)
        state = json.loads(pipeline.read_text())
        self.assertTrue(state["A_outcomes_accessed"])
        self.assertEqual(state["stages"]["freeze"]["status"], "frozen")

    def test_A_is_loaded_only_after_freeze_and_failed_event_can_resume(self):
        original = self.a.read_bytes()
        self.a.unlink()
        with self.assertRaises(FileNotFoundError):
            run_pipeline(self.config, no_plots=True)
        state = json.loads((self.run / "pipeline_manifest.json").read_text())
        self.assertEqual(state["failed_stage"], "A_evaluation")
        self.assertTrue((self.run / "frozen_manifest.json").exists())
        self.a.write_bytes(original)
        with patch("wag.training.run_training", side_effect=AssertionError("No retraining")), patch("wag.training.final_refit", side_effect=AssertionError("No refitting")):
            result = run_pipeline(replace(self.config, resume=True), no_plots=True)
        self.assertEqual(result["status"], "pipeline_complete")
        self.assertEqual(len(list((self.run / "evaluations").iterdir())), 2)

    def test_report_failure_reuses_recorded_A_event(self):
        with patch("wag.reporting.write_report", side_effect=RuntimeError("injected report failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run_pipeline(self.config, no_plots=True)
        state = json.loads((self.run / "pipeline_manifest.json").read_text())
        event = state["stages"]["evaluation"]["evaluation_dir"]
        with patch("wag.evaluation.evaluate", side_effect=AssertionError("A event must be reused")):
            result = run_pipeline(replace(self.config, resume=True), no_plots=True)
        self.assertEqual(result["evaluation_dir"], event)
        self.assertEqual(result["status"], "pipeline_complete")

    def test_changed_config_code_A_or_corrupted_state_refuse_resume(self):
        run_pipeline(self.config, no_plots=True)
        with self.assertRaisesRegex(ValueError, "identical"):
            run_pipeline(replace(self.config, resume=True, seed=43), no_plots=True)
        def changed_source(path):
            result = file_fingerprint(path)
            if str(path).endswith("wag/workflow.py"):
                result["sha256"] = "changed source"
            return result
        with patch("wag.data.file_fingerprint", side_effect=changed_source):
            with self.assertRaisesRegex(ValueError, "identical"):
                run_pipeline(replace(self.config, resume=True), no_plots=True)
        settings = Path(self.config.classical_config)
        original_settings = settings.read_bytes()
        settings.write_bytes(original_settings + b"\n")
        with self.assertRaisesRegex(ValueError, "identical"):
            run_pipeline(replace(self.config, resume=True), no_plots=True)
        settings.write_bytes(original_settings)
        with self.a.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "A CSV changed"):
            run_pipeline(replace(self.config, resume=True), no_plots=True)
        manifest = self.run / "pipeline_manifest.json"
        state = json.loads(manifest.read_text())
        state["status"] = "invented"
        manifest.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "checksum"):
            run_pipeline(replace(self.config, resume=True), no_plots=True)

    def test_no_overwrite_or_concurrent_pipeline(self):
        self.run.mkdir()
        with self.assertRaises(FileExistsError):
            run_pipeline(self.config, no_plots=True)
        self.run.rmdir()
        lock = self.run.parent / ("." + self.run.name + ".pipeline.lock")
        lock.write_text("existing owner")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            run_pipeline(self.config, no_plots=True)
        self.assertEqual(lock.read_text(), "existing owner")
        self.assertFalse(self.run.exists())

    def test_full_synthetic_loading_to_report_and_plots(self):
        from matplotlib.figure import Figure
        original = Figure.savefig
        def small(figure, *args, **kwargs):
            kwargs["dpi"] = 150
            return original(figure, *args, **kwargs)
        with patch.object(Figure, "savefig", small):
            result = run_pipeline(self.config, _asset_root=self.assets())
        self.assertEqual(result["status"], "pipeline_complete")
        images = list((Path(result["report_dir"]) / "figures").glob("*.png"))
        self.assertEqual(len(images), 4)
        self.assertTrue(all(path.stat().st_size > 1000 for path in images))

    def test_cli_full_classical_run_and_no_numeric_workflow_import(self):
        from wag.config import PROJECT_ROOT
        process = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "run", "--no-plots",
                 "--run-dir", str(self.run), "--train-csv", str(self.b), "--test-csv", str(self.a),
                 "--classical-config", self.config.classical_config, "--model-scope", "classical", "--workers", "1", "--device", "cpu"],
                 cwd=self.root, capture_output=True, text=True, env=os.environ.copy())
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('"pipeline_complete"', process.stdout)
        self.assertTrue((self.run / "pipeline_manifest.json").exists())
        code = "import sys; import wag.workflow; assert not set(sys.modules)&{'numpy','sklearn','torch','joblib'}"
        check = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_full_neural_CPU_workflow_when_PyTorch_is_available(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch unavailable; actual full neural workflow acceptance pending")
        from wag.neural import load_neural_settings
        from test_data import records, write_csv
        settings = load_neural_settings()
        settings.update(hidden_layers=[4], max_epochs=2)
        settings["constraints"]["collocation_count"] = 4
        path = self.root / "neural.json"
        path.write_text(json.dumps(settings))
        for file, family, count in ((self.b, "B", 30), (self.a, "A", 12)):
            rows = records(count, family=family)
            for i, row in enumerate(rows):
                row["hcpvi_gas_frac"] = .01 + i * .005
            write_csv(file, rows)
        config = replace(self.config, model_scope="neural", neural_config=str(path))
        result = run_pipeline(config, no_plots=True)
        self.assertEqual(result["status"], "pipeline_complete")
        self.assertIn(result["primary_model_id"], ("NN_MLP", "NN_PINN_HARD"))

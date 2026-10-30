"""Synthetic report/plot acceptance; no official dataset fits or A scoring."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from wag.config import Config, PROJECT_ROOT
from wag.evaluation import evaluate
from wag.models.classical import load_classical_settings
from wag.reporting import load_results, validation_importance, verify_figure_assets, write_report
from wag.runtime import THREAD_ENV
from wag.training import final_refit, run_training
from test_data import records, write_csv


class ReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.b, self.a = self.root / "B.csv", self.root / "A.csv"
        write_csv(self.b, records(30))
        write_csv(self.a, records(12, family="A"))
        settings = load_classical_settings()
        settings.update(enabled_models=["Ridge"], variants=["plain"])
        path = self.root / "classical.json"
        path.write_text(json.dumps(settings))
        self.run = self.root / "run"
        self.config = Config(train_csv=str(self.b), test_csv=str(self.a), run_dir=str(self.run),
                             classical_config=str(path), model_scope="classical", device="cpu", workers=1, inner_folds=2, outer_folds=3)
        limits = patch.dict(os.environ, {name: "1" for name in THREAD_ENV})
        limits.start()
        self.addCleanup(limits.stop)

    def train(self):
        self.assertEqual(run_training(self.config)["status"], "validation_complete")

    def test_holdout_report_truthful_exports_no_fitting_or_A_reads(self):
        self.train()
        self.b.unlink()
        self.a.unlink()
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No fitting")):
            result = write_report(self.run)
        directory = Path(result["report_dir"])
        text = (directory / "report.txt").read_text()
        self.assertIn("Testing evaluation: not supplied", text)
        self.assertIn("not an unbiased", text)
        self.assertFalse((directory / "cv_metrics.csv").exists())
        importance = pd.read_csv(directory / "feature_importance.csv")
        self.assertTrue(importance.importance_mean.max() > 0)
        self.assertAlmostEqual(importance.loc[importance.feature == "depth_ft", "importance_mean"].iloc[0], 0)
        records = pd.read_csv(directory / "importance_records.csv")
        self.assertEqual(records.n.unique().tolist(), [6])
        self.assertEqual(len(records), 55)
        r2 = records.groupby("feature").importance_R2.mean()
        rmse = records.groupby("feature").importance.mean()
        np.testing.assert_allclose(importance.set_index("feature").importance_mean.sort_index(), r2.sort_index())
        rmse_export = pd.read_csv(directory / "feature_importance_rmse.csv").set_index("feature")
        np.testing.assert_allclose(rmse_export.importance_mean.sort_index(), rmse.sort_index())
        self.assertTrue(pd.read_csv(directory / "metrics.csv").split.eq("B_holdout_validation").all())

    def test_nested_report_separates_pooled_and_mean_fold_metrics(self):
        self.config = replace(self.config, validation="nested-cv")
        self.train()
        result = write_report(self.run)
        directory = Path(result["report_dir"])
        cv = pd.read_csv(directory / "cv_metrics.csv")
        actual = pd.read_csv(self.run / "validation_metrics.csv")
        np.testing.assert_allclose(cv.RMSE, actual.mean_outer_RMSE)
        self.assertEqual(len(pd.read_csv(directory / "importance_records.csv")), 165)
        self.assertIn("Pooled OOF metrics differ", (directory / "report.txt").read_text())

    def test_test_event_linkage_and_saved_primary_stay_fixed(self):
        self.train()
        final_refit(self.config)
        event = evaluate(self.config)
        directory = Path(write_report(self.run, evaluation_dir=event["evaluation_dir"])["report_dir"])
        text = (directory / "report.txt").read_text()
        self.assertIn("not observed field", text)
        self.assertNotRegex(text, r"\b[AB]\b")
        self.assertIn("Final refit: 30 Training/Validation rows", text)
        self.assertIn("A_external_simulation", pd.read_csv(directory / "metrics.csv").split.tolist())
        state = json.loads((Path(event["evaluation_dir"]) / "evaluation_manifest.json").read_text())
        state["primary_model_id"] = "wrong"
        (Path(event["evaluation_dir"]) / "evaluation_manifest.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            write_report(self.run, evaluation_dir=event["evaluation_dir"])

    def test_corruption_failed_validation_and_existing_directory_rejected(self):
        self.train()
        output = self.root / "report"
        write_report(self.run, report_dir=output)
        with self.assertRaises(FileExistsError):
            write_report(self.run, report_dir=output)
        with (self.run / "validation_predictions.csv").open("a") as stream:
            stream.write("corrupt")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            write_report(self.run)

    def test_importance_recomputes_dependencies_and_is_seeded_B_only(self):
        from wag.features import make_model_pipeline
        from sklearn.linear_model import Ridge
        frame = pd.read_csv(self.b)
        model = make_model_pipeline(Ridge())
        model.fit(frame.iloc[:20], frame.rf_incremental_pct.iloc[:20])
        kwargs = dict(kind="classical", model_id="Ridge", seed=42, fold=0)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No importance fitting")):
            first = validation_importance(model, frame.iloc[20:], **kwargs)
            second = validation_importance(model, frame.iloc[20:], **kwargs)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "B outer"):
            validation_importance(model, pd.read_csv(self.a), **kwargs)

    def assets(self):
        # Render QA uses an explicit temporary font fixture because the connector
        # cannot download the repository's binary Times font. Production has no fallback.
        from matplotlib import get_data_path
        root = self.root / "assets"
        (root / "figures").mkdir(parents=True)
        (root / "font_files").mkdir()
        (root / "configs").mkdir()
        for relative in ("figures/common.py", "figures/figure_6.py", "figures/figure_7.py"):
            shutil.copy(PROJECT_ROOT / relative, root / relative)
        shutil.copy(Path(get_data_path()) / "fonts/ttf/DejaVuSerif.ttf", root / "font_files/times.ttf")
        hashes = {}
        for path in list((root / "figures").iterdir()) + list((root / "font_files").iterdir()):
            data = path.read_bytes()
            hashes[str(path.relative_to(root))] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        (root / "configs/figure_integrity.json").write_text(json.dumps({"git_blob_hashes": hashes}))
        return root

    def test_saved_result_rendering_uses_existing_style_and_no_refits(self):
        from wag.plotting import plot_saved_results
        self.train()
        assets = self.assets()
        self.b.unlink()
        self.a.unlink()
        from matplotlib.figure import Figure
        original_save = Figure.savefig
        def small_fixture_image(figure, *args, **kwargs):
            kwargs["dpi"] = 150  # bounded test memory; production keeps preserved 600 dpi
            return original_save(figure, *args, **kwargs)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No plot fitting")), patch.object(Figure, "savefig", small_fixture_image):
            result = plot_saved_results(self.run, _asset_root=assets, report_dir=self.root / "render")
        self.assertEqual(result["status"], "plots_complete")
        self.assertEqual(len(result["plots"]), 2)
        from PIL import Image
        for row in result["plots"]:
            image = Image.open(Path(result["report_dir"]) / row["path"])
            self.assertGreater(image.width, 1000)
            self.assertGreater(image.height, 750)
        verify_figure_assets(assets)
        (assets / "figures/figure_6.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            plot_saved_results(self.run, _asset_root=assets)

    def test_original_font_preservation_when_available(self):
        if not (PROJECT_ROOT / "font_files/times.ttf").exists():
            self.skipTest("Binary repository Times font unavailable to connector; font render acceptance pending checkout")
        verify_figure_assets()

    def test_report_CLI_needs_no_GPU_data_or_PyTorch(self):
        self.train()
        self.b.unlink()
        self.a.unlink()
        process = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "report", "--run-dir", str(self.run)],
                                 cwd=self.root, capture_output=True, text=True, env=os.environ.copy())
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "report_complete")


if __name__ == "__main__":
    unittest.main()

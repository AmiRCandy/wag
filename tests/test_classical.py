"""Small synthetic fits only: registry, guided weighting and bounded CPU work."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_info

from test_features import fixture
from wag.config import Config
from wag.cpu import fit_models, search_models
from wag.features import RawFeatures, make_preprocessor, preprocessing_metadata
from wag.models.classical import (FAMILIES, GuidedRegressor, Registry, assert_cpu_estimator,
                                  build_registry, load_classical_settings, raw_anchors)
from wag.runtime import THREAD_ENV, cpu_parallelism


def settings_for(*families, variants=("plain", "guided")):
    settings = load_classical_settings()
    settings["enabled_models"] = list(families)
    settings["variants"] = list(variants)
    settings["guided"]["anchor_count"] = 4
    return settings


class ProbeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_jobs=1, random_state=42, offset=0, directory=None, fail=False):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.offset = offset
        self.directory = directory
        self.fail = fail

    def fit(self, X, y, sample_weight=None):
        if self.fail:
            raise TypeError("Intentional fit error; never retry without weights")
        self.matrix_ = np.asarray(X).copy()
        self.targets_ = np.asarray(y).copy()
        self.weights_ = None if sample_weight is None else np.asarray(sample_weight).copy()
        self.mean_ = float(np.average(y, weights=sample_weight))
        if self.directory:
            import uuid
            start = time.time()
            time.sleep(0.15)
            record = {"pid": os.getpid(), "start": start, "end": time.time(),
                      "env": {key: os.environ.get(key) for key in THREAD_ENV},
                      "threads": [pool["num_threads"] for pool in threadpool_info()]}
            (Path(self.directory) / (uuid.uuid4().hex + ".json")).write_text(json.dumps(record))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_ + self.offset)


class ClassicalTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {key: "1" for key in THREAD_ENV})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.X = fixture(18)
        self.y = self.X.rf_incremental_pct
        self.config = Config(workers=2, inner_folds=2, search_candidates=2)

    def test_registry_preserves_families_defaults_and_CPU_policy(self):
        settings = load_classical_settings()
        self.assertEqual(tuple(settings["enabled_models"]), FAMILIES)
        settings["enabled_models"] = list(FAMILIES[:9])
        registry = build_registry(settings)
        self.assertEqual(set(registry.estimators), set(FAMILIES[:9]))
        for estimator in registry.estimators.values():
            assert_cpu_estimator(estimator)
        self.assertEqual(registry.estimators["RandomForest"].n_estimators, 500)
        self.assertEqual(len(settings["models"]["ExtraTrees"]["search"]["n_estimators"]), 3)
        self.assertEqual(settings["models"]["LightGBM"]["defaults"]["subsample_freq"], 1)

    def test_configuration_blocks_resource_seed_GPU_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            for name in ("n_jobs", "device", "random_state"):
                settings = load_classical_settings()
                settings["models"]["RandomForest"]["search"][name] = [8]
                path.write_text(json.dumps(settings))
                with self.assertRaisesRegex(ValueError, "runtime policy"):
                    load_classical_settings(path)
        with self.assertRaises(ValueError):
            assert_cpu_estimator(ProbeRegressor(n_jobs=-1))

    def test_missing_optional_models_error_or_explicit_skip(self):
        settings = settings_for("XGBoost", "LightGBM")
        with patch("wag.models.classical.dependency_preflight", side_effect=RuntimeError("missing package")):
            with self.assertRaisesRegex(RuntimeError, "missing package"):
                build_registry(settings)
            settings["missing_model_policy"] = "skip"
            results = search_models(self.X, self.y, self.config, settings)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.status == "skipped" for result in results.values()))
        self.assertTrue(all("missing package" in result.record()["reason"] for result in results.values()))

    def test_all_sklearn_families_small_plain_and_supported_guided_fits(self):
        settings = settings_for(*FAMILIES[:9])
        settings["models"]["RandomForest"]["defaults"]["n_estimators"] = 3
        settings["models"]["ExtraTrees"]["defaults"]["n_estimators"] = 3
        settings["models"]["GradientBoosting"]["defaults"]["n_estimators"] = 3
        results = fit_models(self.X, self.y, self.config, settings)
        for model_id, result in results.items():
            if model_id == "KNN_GUIDED":
                self.assertEqual(result.status, "skipped")
                continue
            self.assertEqual(result.status, "fitted", result.record())
            prediction = result.estimator.predict(self.X.iloc[:3])
            self.assertTrue(np.isfinite(prediction).all())
            self.assertFalse(result.details["fallback_performed"])
        self.assertEqual(load_classical_settings()["models"]["RandomForest"]["defaults"]["n_estimators"], 500)

    def test_weighted_guidance_applies_weights_and_recomputes_raw_dependencies(self):
        model = GuidedRegressor(ProbeRegressor(), anchor_count=4).fit(self.X, self.y)
        self.assertEqual(model.estimator_.weights_.tolist(), [1] * 18 + [0.3] * 4)
        self.assertEqual(model.estimator_.matrix_.shape, (22, 17))
        self.assertEqual(model.fit_record_["preprocessing_fit_rows"], 18)
        self.assertFalse(model.fit_record_["permeability_monotonicity"])
        reference = preprocessing_metadata(make_preprocessor().fit(self.X))
        self.assertEqual(preprocessing_metadata(model.preprocessor_), reference)
        anchors, labels = raw_anchors(self.X, self.y.to_numpy(), injection_mode="volumes", count=4, seed=42)
        np.testing.assert_array_equal(model.estimator_.matrix_[18:], model.preprocessor_.transform(anchors))
        np.testing.assert_array_equal(model.estimator_.targets_[18:], labels)
        self.assertNotIn("rf_incremental_pct", anchors)
        self.assertEqual(labels[:2].tolist(), [float(np.percentile(self.y, 10))] * 2)
        self.assertEqual(labels[2:].tolist(), [float(np.percentile(self.y, 90))] * 2)

    def test_guided_clipping_is_separate_from_raw_predictions_and_serializes(self):
        model = GuidedRegressor(ProbeRegressor(offset=1000), anchor_count=4).fit(self.X, self.y)
        self.assertTrue((model.predict_raw(self.X) > 100).all())
        self.assertEqual(model.predict(self.X).tolist(), [100] * 18)
        self.assertFalse(hasattr(clone(model), "estimator_"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "guided.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict_raw(self.X), model.predict_raw(self.X))
            self.assertEqual(restored.fit_record_, model.fit_record_)

    def test_unsupported_weighted_variant_is_never_unweighted_fallback(self):
        results = search_models(self.X, self.y, self.config, settings_for("KNN"))
        self.assertEqual(results["KNN"].status, "fitted")
        self.assertEqual(results["KNN_GUIDED"].status, "skipped")
        self.assertIn("no unweighted fallback", results["KNN_GUIDED"].record()["reason"])

    def test_actual_search_uses_raw_fold_local_pipeline_and_RMSE(self):
        settings = settings_for("Ridge", variants=("plain",))
        folds = [(np.arange(9), np.arange(9, 18)), (np.arange(9, 18), np.arange(9))]
        observed = []
        original = RawFeatures.fit
        def spy(raw, X, y=None):
            observed.append(tuple(X.index))
            return original(raw, X, y)
        with patch.object(RawFeatures, "fit", spy):
            results = search_models(self.X, self.y, Config(workers=1, inner_folds=2, search_candidates=2), settings,
                                    cv=folds, search_spaces={"Ridge": {"alpha": [0.1, 1.0]}})
        result = results["Ridge"]
        self.assertEqual(result.status, "fitted", result.record())
        self.assertEqual(observed.count(tuple(range(9))), 2)
        self.assertEqual(observed.count(tuple(range(9, 18))), 2)
        self.assertEqual(observed.count(tuple(range(18))), 1)
        self.assertEqual(result.details["score_role"], "inner_tuning_only")
        self.assertEqual(result.details["actual_candidates"], 2)
        for trial in result.details["trials"]:
            self.assertAlmostEqual(trial["mean_inner_rmse"], np.mean(trial["fold_inner_rmse"]))
        json.dumps(result.record(), allow_nan=False)

    def test_guided_search_anchors_are_inside_each_fit(self):
        settings = settings_for("Ridge", variants=("guided",))
        results = search_models(self.X, self.y, self.config, settings, search_spaces={"Ridge": {"alpha": [0.1, 1.0]}})
        result = results["Ridge_GUIDED"]
        self.assertEqual(result.status, "fitted", result.record())
        self.assertEqual(result.estimator.preprocessor_.named_steps["raw"].n_fit_rows_, 18)
        self.assertEqual(result.details["guidance"]["anchor_rows"], 4)
        self.assertEqual(result.details["pre_dispatch"], 2)

    def test_full_grid_is_explicit_and_randomized_budget_is_bounded(self):
        settings = settings_for("Ridge", variants=("plain",))
        space = {"Ridge": {"alpha": [0.1, 1, 2]}}
        random = search_models(self.X, self.y, self.config, settings, search_spaces=space)["Ridge"]
        grid = search_models(self.X, self.y, Config(workers=1, search="grid", inner_folds=2, search_candidates=2), settings, search_spaces=space)["Ridge"]
        self.assertEqual(random.details["actual_candidates"], 2)
        self.assertEqual(grid.details["actual_candidates"], 3)
        self.assertEqual(Config().search_candidates, 20)

    def test_bad_CV_and_resource_dimensions_produce_visible_failed_records(self):
        settings = settings_for("Ridge", variants=("plain",))
        bad = [(np.arange(12), np.arange(8, 18)), (np.arange(8, 18), np.arange(12))]
        result = search_models(self.X, self.y, self.config, settings, cv=bad,
                               search_spaces={"Ridge": {"alpha": [1]}})["Ridge"]
        self.assertEqual(result.status, "failed")
        self.assertIn("overlap", result.details["error"]["message"])
        result = search_models(self.X, self.y, self.config, settings,
                               search_spaces={"Ridge": {"n_jobs": [8]}})["Ridge"]
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.details["fallback_performed"])

    def test_fit_error_keeps_type_message_traceback_and_never_retries(self):
        settings = settings_for("Ridge", variants=("guided",))
        registry = Registry({"Ridge": ProbeRegressor(fail=True)}, {}, settings)
        with patch("wag.cpu.build_registry", return_value=registry):
            result = search_models(self.X, self.y, self.config, settings)["Ridge_GUIDED"]
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.details["error"]["type"], "TypeError")
        self.assertIn("Intentional fit error", result.details["error"]["traceback"])
        self.assertIsNone(result.estimator)

    def test_fitting_order_does_not_change_task_seeds_or_predictions(self):
        one = settings_for("Ridge", "LinearRegression", variants=("plain",))
        two = deepcopy(one)
        two["enabled_models"].reverse()
        first = fit_models(self.X, self.y, self.config, one)
        second = fit_models(self.X, self.y, self.config, two)
        for name in first:
            self.assertEqual(first[name].details["seed"], second[name].details["seed"])
            np.testing.assert_array_equal(first[name].estimator.predict(self.X), second[name].estimator.predict(self.X))

    def test_real_search_concurrency_and_native_thread_caps(self):
        settings = settings_for("Ridge", variants=("plain",))
        with tempfile.TemporaryDirectory() as temporary:
            registry = Registry({"Ridge": ProbeRegressor(directory=temporary)}, {}, settings)
            with patch("wag.cpu.build_registry", return_value=registry):
                result = search_models(self.X, self.y, self.config, settings,
                                       search_spaces={"Ridge": {"offset": [0, 1]}})["Ridge"]
            self.assertEqual(result.status, "fitted", result.record())
            records = [json.loads(path.read_text()) for path in Path(temporary).glob("*.json")]
            self.assertEqual(len(records), 5)  # two candidates * two folds + coordinator refit
            peak = max(sum(other["start"] <= record["start"] < other["end"] for other in records) for record in records)
            self.assertLessEqual(peak, 2)
            for record in records:
                self.assertTrue(all(value == "1" for value in record["env"].values()))
                self.assertTrue(all(value == 1 for value in record["threads"]))

    def test_nested_pools_are_rejected_and_context_can_be_reused(self):
        with cpu_parallelism(2):
            with self.assertRaisesRegex(RuntimeError, "Nested"), cpu_parallelism(2):
                pass
        with cpu_parallelism(2):
            pass

    def test_real_workers_cannot_create_another_pool(self):
        from joblib import Parallel, delayed
        def worker():
            try:
                with cpu_parallelism(2):
                    pass
            except RuntimeError as exc:
                return str(exc)
            return "not rejected"
        with cpu_parallelism(2):
            values = Parallel(n_jobs=2)(delayed(worker)() for _ in range(2))
        self.assertTrue(all("Nested" in value for value in values))

    def test_A_is_rejected_and_target_index_alignment_is_checked(self):
        X = self.X.copy()
        X["reservoir_family"] = "A"
        result = search_models(X, self.y, self.config, settings_for("Ridge", variants=("plain",)))["Ridge"]
        self.assertEqual(result.status, "failed")
        self.assertIn("reservoir_family=B", result.details["error"]["message"])
        with self.assertRaisesRegex(ValueError, "index"):
            fit_models(self.X, self.y.iloc[::-1], self.config, settings_for("Ridge"))

    def optional_package_smoke(self, family):
        settings = settings_for(family)
        settings["models"][family]["defaults"]["n_estimators"] = 3
        try:
            registry = build_registry(settings)
        except (ImportError, RuntimeError, OSError) as exc:
            self.skipTest(f"{family} unavailable in this environment: {exc}")
        assert_cpu_estimator(registry.estimators[family])
        results = fit_models(self.X, self.y, self.config, settings)
        for result in results.values():
            self.assertEqual(result.status, "fitted", result.record())
            self.assertTrue(np.isfinite(result.estimator.predict(self.X)).all())

    def test_XGBoost_when_available(self):
        self.optional_package_smoke("XGBoost")

    def test_LightGBM_when_available(self):
        self.optional_package_smoke("LightGBM")


if __name__ == "__main__":
    unittest.main()

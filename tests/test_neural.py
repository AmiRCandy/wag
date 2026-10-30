"""Synthetic neural contracts; tensor execution is conditional on installed torch."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from wag.config import Config, PROJECT_ROOT
from wag.features import FeatureSpec, make_preprocessor, preprocessing_metadata
from wag.neural import (NeuralResult, collocation_raw, fit_neural, fit_neural_models,
                        imputed_raw, inner_partition, load_neural_bundle,
                        load_neural_settings, validate_neural_settings)
from wag.runtime import THREAD_ENV, cpu_parallelism, neural_execution, task_seed
from test_features import fixture


HAS_TORCH = importlib.util.find_spec("torch") is not None


class NeuralContractTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_neural_settings()
        self.X = fixture()
        self.env = patch.dict(os.environ, {key: "1" for key in THREAD_ENV})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_defaults_and_settings_round_trip(self):
        self.assertEqual(self.settings["hidden_layers"], [96, 96, 48])
        self.assertEqual(self.settings["constraints"]["collocation_count"], 3000)
        self.assertEqual(validate_neural_settings(json.loads(json.dumps(self.settings))), self.settings)

    def test_invalid_settings_reject_silent_options(self):
        alternatives = [("max_epochs", True), ("learning_rate", 0), ("weight_decay", float("nan")),
                        ("hidden_layers", [True]), ("enabled_models", ["NN_MLP", "NN_MLP"]),
                        ("bounds", [-25, -30]), ("bounds", [-101, 100]), ("unknown", 2)]
        for key, value in alternatives:
            settings = deepcopy(self.settings)
            settings[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_neural_settings(settings)
        settings = deepcopy(self.settings)
        settings["constraints"]["permeability_monotonicity_weight"] = 0.5
        with self.assertRaises(ValueError):
            validate_neural_settings(settings)

    def test_common_inner_split_and_invalid_positions(self):
        seed = task_seed(42, "split", "neural_inner", 0)
        train, valid = inner_partition(len(self.X), self.settings, seed)
        self.assertEqual(len(valid), 4)
        self.assertFalse(set(train) & set(valid))
        a, b = inner_partition(len(self.X), self.settings, seed)
        np.testing.assert_array_equal(a, train)
        np.testing.assert_array_equal(b, valid)
        for supplied in (([0, 1], [1, 2]), ([0, 1], [2]), ([0, 0], list(range(2, 18))),
                         ([0.0, 1.0], list(range(2, 18))), (list(range(18)), [])):
            with self.assertRaises(ValueError):
                inner_partition(len(self.X), self.settings, seed, supplied)

    def test_collocation_recomputes_from_canonical_training_rows(self):
        preprocessor = make_preprocessor().fit(self.X)
        raw = imputed_raw(preprocessor, self.X)
        generated = collocation_raw(raw, 40, 5)
        self.assertEqual(generated.shape, (40, 11))
        np.testing.assert_array_equal(generated, collocation_raw(raw, 40, 5))
        self.assertTrue(np.all(generated[:, -2] >= raw[:, -2].min()))
        self.assertTrue(np.all(generated[:, -2] <= raw[:, -2].max()))
        nongas = [i for i in range(11) if i != 9]
        originals = {tuple(row[nongas]) for row in raw}
        self.assertTrue(all(tuple(row[nongas]) in originals for row in generated))
        self.assertEqual(collocation_raw(raw, 0, 5).shape, (0, 11))

    def test_A_and_target_misalignment_rejected_before_torch(self):
        config = Config(device="cpu")
        with self.assertRaisesRegex(ValueError, "B simulation"):
            fit_neural(fixture(family="A"), self.X.rf_incremental_pct, config, self.settings, "NN_MLP")
        with self.assertRaisesRegex(ValueError, "align"):
            fit_neural(self.X, self.X.rf_incremental_pct.iloc[::-1], config, self.settings, "NN_MLP")

    def test_fixed_epoch_contract(self):
        config = Config(device="cpu")
        for epochs in (True, 0, 1001):
            with self.assertRaises(ValueError):
                fit_neural(self.X, self.X.rf_incremental_pct, config, self.settings, "NN_MLP", epochs=epochs)
        with self.assertRaises(ValueError):
            fit_neural(self.X, self.X.rf_incremental_pct, config, self.settings, "NN_MLP", epochs=2, inner_split=([0, 1], [2]))
        with self.assertRaises(ValueError):
            fit_neural_models(self.X, self.X.rf_incremental_pct, config, self.settings, selected_epochs={"NN_MLP": 2})
        with patch("wag.neural.fit_neural") as fit, self.assertRaises(ValueError):
            fit_neural_models(self.X, self.X.rf_incremental_pct, config, self.settings,
                              selected_epochs={"NN_MLP": 2, "NN_PINN_HARD": 1001})
        fit.assert_not_called()

    def test_models_dispatched_sequentially_with_selected_budgets(self):
        calls = []
        def fit(*args, **kwargs):
            calls.append((args[4], kwargs["epochs"]))
            return NeuralResult(args[4], "fitted")
        with patch("wag.neural.fit_neural", side_effect=fit):
            results = fit_neural_models(self.X, self.X.rf_incremental_pct, Config(device="cpu"), self.settings,
                                        selected_epochs={"NN_MLP": 3, "NN_PINN_HARD": 5})
        self.assertEqual(calls, [("NN_MLP", 3), ("NN_PINN_HARD", 5)])
        self.assertEqual([r.model_id for r in results], self.settings["enabled_models"])

    def test_neural_blocked_inside_CPU_pool_before_dependency_access(self):
        with cpu_parallelism(1), patch("wag.runtime.dependency_preflight") as dependencies:
            with self.assertRaisesRegex(RuntimeError, "coordinator"):
                with neural_execution("cpu"):
                    pass
            dependencies.assert_not_called()

    def test_neural_context_blocks_nested_and_CPU_pool_and_recovers(self):
        with patch("wag.runtime.dependency_preflight"), patch("wag.runtime.configure_torch", return_value={"selected": "cpu", "torch_threads": 1, "torch_interop_threads": 1}):
            with self.assertRaisesRegex(ValueError, "test interruption"):
                with neural_execution("cpu"):
                    with self.assertRaisesRegex(RuntimeError, "coordinator"):
                        with neural_execution("cpu"):
                            pass
                    with self.assertRaisesRegex(RuntimeError, "Nested"):
                        with cpu_parallelism(1):
                            pass
                    raise ValueError("test interruption")
            with neural_execution("cpu"):
                pass

    def test_neural_rejects_changed_torch_thread_settings(self):
        with patch("wag.runtime.dependency_preflight"), patch("wag.runtime.configure_torch", return_value={"torch_threads": 8, "torch_interop_threads": 1}):
            with self.assertRaisesRegex(RuntimeError, "thread settings changed"):
                with neural_execution("cpu"):
                    pass

    def test_neural_rejects_worker_and_nonmain_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        with patch("multiprocessing.current_process") as process:
            process.return_value.name = "LokyProcess-1"
            with self.assertRaisesRegex(RuntimeError, "coordinator"):
                with neural_execution("cpu"):
                    pass
        def operation():
            with neural_execution("cpu"):
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaisesRegex(RuntimeError, "coordinator"):
                pool.submit(operation).result()

    def test_missing_torch_explicit_no_install_or_fallback(self):
        with patch("wag.runtime.dependency_preflight", side_effect=RuntimeError("Missing dependencies: torch")):
            with self.assertRaisesRegex(RuntimeError, "Missing dependencies"):
                fit_neural(self.X, self.X.rf_incremental_pct, Config(device="cpu"), self.settings, "NN_MLP")

    def test_safe_import_and_settings_require_no_torch_workers_or_IO(self):
        code = '''
from wag.runtime import bootstrap_threads
bootstrap_threads()
import multiprocessing, pathlib, sys, tempfile, os
with tempfile.TemporaryDirectory() as directory:
    os.chdir(directory)
    import wag.neural
    assert 'torch' not in sys.modules
    assert not multiprocessing.active_children()
    assert list(pathlib.Path(directory).iterdir()) == []
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed; neural tensor execution not verified")
class NeuralTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.update({key: "1" for key in THREAD_ENV})
        from wag.runtime import configure_torch
        configure_torch("cpu")
        import torch
        cls.torch = torch

    def setUp(self):
        self.X = fixture()
        self.settings = load_neural_settings()
        self.settings.update(hidden_layers=[8, 4], max_epochs=4)
        self.settings["constraints"]["collocation_count"] = 7
        self.settings["early_stopping"]["patience"] = 2
        self.config = Config(device="cpu", workers=8)

    def fit(self, name="NN_MLP", **kwargs):
        result = fit_neural(self.X, self.X.rf_incremental_pct, self.config, self.settings, name, **kwargs)
        self.assertEqual(result.status, "fitted", result.record())
        json.dumps(result.record(), allow_nan=False)
        return result

    def test_CPU_fits_both_families_and_bounded_negative_predictions(self):
        from wag.models.neural import HardPINN
        for name in ("NN_MLP", "NN_PINN_HARD"):
            result = self.fit(name)
            predictions = result.estimator.predict(self.X)
            self.assertEqual(predictions.shape, (18,))
            self.assertTrue(np.isfinite(predictions).all())
            self.assertEqual(self.torch.get_num_threads(), 1)
            self.assertEqual(self.torch.get_num_interop_threads(), 1)
            self.assertEqual(result.details["device"]["selected"], "cpu")
            self.assertFalse(result.details["constraints"]["whole_network_monotonicity_guaranteed"])
        network = HardPINN(2, bounds=(-100, 100), hidden=(3,))
        with self.torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.trunk[-1].bias.fill_(-1)
            values = network(self.torch.randn(10, 2))
        self.assertTrue((values < 0).all())
        self.assertTrue((values >= -100).all() and (values <= 100).all())

    def test_differentiable_map_matches_all_modes_missing_values_and_constants(self):
        from wag.models.neural import TorchFeatures
        for mode in ("volumes", "design", "rates"):
            X = self.X.copy()
            X.loc[2, "effective_perm_md"] = np.nan
            X["oil_visc_mult"] = 1.2
            preprocessing = make_preprocessor(mode).fit(X)
            raw = imputed_raw(preprocessing, X)
            mapping = TorchFeatures(preprocessing, raw)
            values = self.torch.tensor(raw, dtype=self.torch.float32)
            coordinates = (values - mapping.raw_mean) / mapping.raw_scale
            actual = mapping(coordinates).detach().numpy()
            np.testing.assert_allclose(actual, preprocessing.transform(X), rtol=2e-5, atol=2e-5)

    def test_gas_derivatives_follow_log_dependency(self):
        from wag.models.neural import TorchFeatures, gas_derivatives
        preprocessor = make_preprocessor().fit(self.X)
        raw = imputed_raw(preprocessor, self.X)
        mapping = TorchFeatures(preprocessor, raw).double()
        z = ((self.torch.tensor(raw, dtype=self.torch.float64) - mapping.raw_mean) / mapping.raw_scale).detach().requires_grad_(True)
        column = preprocessing_metadata(preprocessor)["active_features"].index("log1p_hcpvi_gas_frac")
        predict = lambda coordinates: mapping(coordinates)[:, column] * mapping.scale[column] + mapping.mean[column]
        first, second = gas_derivatives(predict, z, 9)
        scale = mapping.raw_scale[9].item()
        np.testing.assert_allclose(first.detach(), scale / (1 + raw[:, 9]), rtol=1e-7)
        np.testing.assert_allclose(second.detach(), -scale**2 / (1 + raw[:, 9])**2, rtol=1e-7)

    def test_analytic_penalty_sign_device_and_finite_parameter_gradients(self):
        from wag.models.neural import shape_penalty
        torch = self.torch
        x = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
        slope = torch.nn.Parameter(torch.tensor(-2.))
        penalty = shape_penalty(lambda z: slope * z[:, 0], x, 0, .5, .3)
        self.assertAlmostEqual(penalty.item(), 2.)
        penalty.backward()
        self.assertTrue(torch.isfinite(slope.grad))
        self.assertEqual(penalty.device, x.device)
        x = x.detach().requires_grad_(True)
        convex = shape_penalty(lambda z: z[:, 0] ** 2, x, 0, .5, .3)
        self.assertAlmostEqual(convex.item(), 1.2, places=5)
        increasing_concave = shape_penalty(lambda z: torch.log1p(z[:, 0]), x, 0, .5, .3)
        self.assertAlmostEqual(increasing_concave.item(), 0.)

    def test_best_checkpoint_restored_and_preprocessing_excludes_inner_holdout(self):
        result = self.fit(inner_split=(np.arange(14), np.arange(14, 18)))
        history = result.details["history"]
        best = min(history, key=lambda row: row["inner_validation_rmse"])
        self.assertEqual(result.details["selected_epochs"], best["epoch"])
        self.assertTrue(result.details["best_checkpoint_restored"])
        prediction = result.estimator.predict(self.X.iloc[14:])
        rmse = np.sqrt(np.mean((prediction - self.X.rf_incremental_pct.iloc[14:].to_numpy()) ** 2))
        self.assertAlmostEqual(rmse, result.details["best_inner_rmse"], places=5)
        metadata = result.details["preprocessing"]
        self.assertEqual(metadata["fit_rows"], 14)
        self.assertEqual(metadata["observed_raw_ranges"]["effective_perm_md"]["max"], 140)

    def test_early_stop_patience_is_used(self):
        self.settings["early_stopping"].update(patience=1, min_delta=1e20)
        result = self.fit()
        self.assertEqual(result.details["epochs_run"], 2)

    def test_nonlast_best_weights_are_restored(self):
        torch = self.torch
        class WorseningOptimizer:
            def __init__(self, parameters, **kwargs):
                self.parameters = list(parameters)
                self.epoch = 0
            def zero_grad(self, **kwargs):
                for parameter in self.parameters:
                    parameter.grad = None
            def step(self):
                self.epoch += 1
                with torch.no_grad():
                    for parameter in self.parameters:
                        parameter.zero_()
                    self.parameters[-1].fill_(50 * self.epoch)
        with patch("torch.optim.Adam", WorseningOptimizer):
            result = self.fit()
        self.assertEqual(result.details["selected_epochs"], 1)
        self.assertEqual(result.details["epochs_run"], 3)
        np.testing.assert_array_equal(result.estimator.predict(self.X), np.full(18, 50.))

    def test_predictions_do_not_require_gradients_and_do_not_change_statistics(self):
        result = self.fit("NN_PINN_HARD")
        metadata = preprocessing_metadata(result.estimator.preprocessor)
        with self.torch.inference_mode():
            prediction = result.estimator.predict(fixture(family="A"))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(metadata, preprocessing_metadata(result.estimator.preprocessor))
        self.assertTrue(all(p.grad is None or self.torch.isfinite(p.grad).all() for p in result.estimator.model.parameters()))

    def test_seed_order_independent_reproducibility(self):
        first = self.fit()
        self.fit("NN_PINN_HARD")
        second = self.fit()
        np.testing.assert_array_equal(first.estimator.predict(self.X), second.estimator.predict(self.X))
        self.assertEqual(first.details["history"], second.details["history"])

    def test_fixed_epoch_refit_no_validation_and_all_passed_B_rows(self):
        result = self.fit(epochs=3, phase="refit")
        self.assertEqual(result.details["selected_epochs"], 3)
        self.assertEqual(result.details["epochs_run"], 3)
        self.assertIsNone(result.details["best_inner_rmse"])
        self.assertIsNone(result.details["inner_validation_positions"])
        self.assertEqual(result.details["preprocessing"]["fit_rows"], 18)

    def test_failure_records_no_CPU_fallback_or_retry(self):
        with patch("wag.neural._fit", side_effect=RuntimeError("CUDA out of memory")) as operation:
            result = fit_neural(self.X, self.X.rf_incremental_pct, self.config, self.settings, "NN_MLP")
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.details["fallback_performed"])
        self.assertIn("out of memory", result.details["error"]["message"])
        operation.assert_called_once()

    def test_portable_state_prediction_round_trip_and_metadata_mismatch(self):
        for name in ("NN_MLP", "NN_PINN_HARD"):
            result = self.fit(name)
            bundle = result.estimator.state_bundle()
            self.assertTrue(all(t.device.type == "cpu" for t in bundle["model_state"].values()))
            self.assertTrue(all(not t.requires_grad for t in bundle["mapping_state"].values()))
            with tempfile.TemporaryDirectory() as directory:
                state_path = Path(directory) / "state.pt"
                preprocessing_path = Path(directory) / "preprocessing.joblib"
                self.torch.save(bundle, state_path)
                joblib.dump(result.estimator.preprocessor, preprocessing_path)
                saved = self.torch.load(state_path, map_location="cpu", weights_only=True)
                restored = load_neural_bundle(saved, joblib.load(preprocessing_path))
                np.testing.assert_array_equal(restored.predict(self.X), result.estimator.predict(self.X))
                # Fresh interpreter: CPU loading does not need the original fit/device.
                code = '''
from wag.runtime import bootstrap_threads
bootstrap_threads()
import joblib, numpy as np, torch, sys
from wag.neural import load_neural_bundle
saved = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
predictor = load_neural_bundle(saved, joblib.load(sys.argv[2]), device='cpu')
X, expected = joblib.load(sys.argv[3])
np.testing.assert_array_equal(predictor.predict(X), expected)
'''
                reference = Path(directory) / "reference.joblib"
                joblib.dump((self.X, result.estimator.predict(self.X)), reference)
                check = subprocess.run([sys.executable, "-B", "-c", code, str(state_path), str(preprocessing_path), str(reference)],
                                       cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(check.returncode, 0, check.stderr)
            different = make_preprocessor().fit(self.X.iloc[:12])
            with self.assertRaisesRegex(ValueError, "metadata differ"):
                load_neural_bundle(bundle, different)

    def test_saved_CPU_state_is_independent_from_live_model(self):
        result = self.fit()
        bundle = result.estimator.state_bundle()
        original = {k: v.clone() for k, v in bundle["model_state"].items()}
        with self.torch.no_grad():
            for parameter in result.estimator.model.parameters():
                parameter.add_(10)
        self.assertTrue(all(self.torch.equal(original[k], v) for k, v in bundle["model_state"].items()))

    def test_explicit_unavailable_cuda_never_falls_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                fit_neural(self.X, self.X.rf_incremental_pct, Config(device="cuda:0"), self.settings, "NN_MLP")

    def test_conditional_GPU_fit_uses_selected_visible_index_and_CPU_loading(self):
        if not self.torch.cuda.is_available():
            self.skipTest("CUDA unavailable; GPU neural execution not verified")
        index = 1 if self.torch.cuda.device_count() > 1 else 0
        device = f"cuda:{index}"
        result = fit_neural(self.X, self.X.rf_incremental_pct, Config(device=device), self.settings, "NN_PINN_HARD", epochs=2)
        self.assertEqual(result.status, "fitted", result.record())
        self.assertEqual(result.details["device"]["logical_index"], index)
        self.assertEqual(str(next(result.estimator.model.parameters()).device), device)
        self.assertTrue(np.isfinite(result.estimator.predict(self.X)).all())
        restored = load_neural_bundle(result.estimator.state_bundle(), result.estimator.preprocessor, device="cpu")
        np.testing.assert_allclose(restored.predict(self.X), result.estimator.predict(self.X), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

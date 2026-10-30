"""Fold isolation, schemas and inference round trips on synthetic B inputs."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.exceptions import NotFittedError

from wag.config import PROJECT_ROOT
from wag.features import (CONTEXT_COLUMNS, FEATURE_VERSION, ConstantFeatures, FeatureSpec,
                          FeatureValidationError, make_model_pipeline, make_preprocessor,
                          preprocessing_metadata)


def fixture(n=18, family="B"):
    rows = []
    for i in range(n):
        rows.append({"source": "OPM_SPE5_simulation", "reservoir_family": family,
                     "effective_perm_md": 10 + 10 * i, "porosity_frac": 0.1 + 0.005 * i,
                     "init_pressure_psi": 2000 + 40 * i, "depth_ft": 4000 + 50 * i,
                     "prod_orat_stb": 9000 + 100 * i, "wag_halfcycle_mo": 3 + i % 4,
                     "waterflood_years": 4 + i % 3, "perm_contrast_exp": 1.1 + i * 0.01,
                     "oil_visc_mult": 1.2 + i * 0.02, "hcpvi_gas_frac": i * 0.05,
                     "wag_ratio_resvol": 0.4 + i * 0.1, "hcpvi_target": 0.1 + i * 0.03,
                     "wag_ratio_target": 1 + i * 0.05, "gas_rate_mscf": 1000 + 500 * i,
                     "wat_rate_stb": 2000 + 300 * i, "mmp_full_psi": 3200,
                     "grid_nx": 7, "grid_ny": 7, "grid_nz": 3, "cell_size_ft": 500,
                     "thickness_mult": 1, "kv_kh_mult": 1, "tl_omega": 0.6, "temperature_F": 160,
                     "injectant": "SPE5 solvent", "process_type": "Miscible",
                     "solvent_fraction": 1, "primary_years": 2, "wag_years": 10,
                     "rf_incremental_pct": -5 + 0.5 * i, "rf_wag_total_pct": 35 + 0.5 * i,
                     "rf_waterflood_total_pct": 40, "perm_mult": (10 + 10 * i) / 250,
                     "porosity_pct": (0.1 + 0.005 * i) * 100})
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_specification_modes_units_and_json_round_trip(self):
        for mode in ("volumes", "design", "rates"):
            spec = FeatureSpec(mode)
            self.assertEqual(FeatureSpec(**json.loads(json.dumps(asdict(spec)))), spec)
            description = spec.description()
            self.assertEqual(set(description["units"]), set(spec.all_features))
            self.assertEqual(len(spec.raw_features), 11)
            self.assertEqual(len(spec.all_features), 17)
            self.assertNotIn("perm_mult", spec.all_features)
            self.assertNotIn("porosity_pct", spec.all_features)
        with self.assertRaises(ValueError):
            FeatureSpec("unknown")
        with self.assertRaises(ValueError):
            FeatureSpec(version="future_version")

    def test_outputs_metadata_names_and_raw_selection(self):
        X = fixture()
        preprocessor = make_preprocessor()
        values = preprocessor.fit_transform(X)
        self.assertEqual(values.shape, (18, 17))
        metadata = preprocessing_metadata(preprocessor)
        self.assertEqual(metadata["specification"]["version"], FEATURE_VERSION)
        self.assertEqual(metadata["fit_rows"], 18)
        self.assertEqual(metadata["active_features"], preprocessor.get_feature_names_out().tolist())
        self.assertNotIn("mmp_full_psi", metadata["active_features"])
        self.assertEqual(metadata["fixed_context"]["mmp_full_psi"], 3200)
        json.dumps(metadata, allow_nan=False)
        np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-12)

    def test_outcomes_equivalents_metadata_and_unknown_columns_cannot_enter(self):
        X = fixture()
        preprocessor = make_preprocessor().fit(X)
        before = preprocessor.transform(X)
        changed = X.copy()
        for name in ("rf_incremental_pct", "rf_wag_total_pct", "rf_waterflood_total_pct", "perm_mult", "porosity_pct"):
            changed[name] = "not_numeric_and_must_be_ignored"
        changed["meta_new"] = np.arange(len(X))
        changed["future_output"] = np.arange(len(X)) * 1000
        np.testing.assert_array_equal(preprocessor.transform(changed), before)
        alternate = make_preprocessor().fit(changed)
        self.assertEqual(preprocessing_metadata(alternate), preprocessing_metadata(preprocessor))

    def test_engineered_formulas_and_recomputed_dependencies(self):
        X = fixture()
        preprocessor = make_preprocessor().fit(X)
        raw = preprocessor.named_steps["raw"].transform(X)
        imputed = preprocessor.named_steps["imputer"].transform(raw)
        engineered = preprocessor.named_steps["engineer"].transform(imputed)
        names = FeatureSpec().all_features
        column = {name: engineered[:, i] for i, name in enumerate(names)}
        np.testing.assert_allclose(column["log10_perm"], np.log10(X.effective_perm_md))
        np.testing.assert_allclose(column["perm_times_porosity"], X.effective_perm_md * X.porosity_frac)
        np.testing.assert_allclose(column["pressure_per_depth"], X.init_pressure_psi / X.depth_ft)
        np.testing.assert_allclose(column["pressure_over_mmp"], X.init_pressure_psi / 3200)
        self.assertEqual(column["log1p_hcpvi_gas_frac"][0], 0)
        changed = X.copy()
        changed["effective_perm_md"] *= 2
        changed_raw = preprocessor.named_steps["raw"].transform(changed)
        new = preprocessor.named_steps["engineer"].transform(preprocessor.named_steps["imputer"].transform(changed_raw))
        np.testing.assert_allclose(new[:, names.index("log10_perm")] - column["log10_perm"], np.log10(2))
        np.testing.assert_allclose(new[:, names.index("perm_times_porosity")], 2 * column["perm_times_porosity"])

    def test_modes_use_only_the_declared_injection_quantities(self):
        X = fixture()
        for mode in ("volumes", "design", "rates"):
            preprocessor = make_preprocessor(mode).fit(X)
            before = preprocessor.transform(X)
            changed = X.copy()
            selected = FeatureSpec(mode).raw_features
            for name in ("gas_rate_mscf", "wat_rate_stb", "hcpvi_target", "wag_ratio_target", "hcpvi_gas_frac", "wag_ratio_resvol"):
                if name not in selected:
                    changed[name] = "must_be_ignored"
            np.testing.assert_array_equal(preprocessor.transform(changed), before)
        self.assertNotIn("total_rate", FeatureSpec("rates").description()["formulas"])

    def test_inconsistent_stage_modes_are_rejected(self):
        inconsistent = make_preprocessor().set_params(raw__injection_mode="rates")
        with self.assertRaisesRegex(FeatureValidationError, "injection mode"):
            inconsistent.fit(fixture())

    def test_validation_and_A_cannot_change_statistics_or_schema(self):
        training = fixture(12)
        training.loc[0, "effective_perm_md"] = np.nan
        training["wag_halfcycle_mo"] = 3
        preprocessor = make_preprocessor().fit(training)
        snapshot = json.dumps(preprocessing_metadata(preprocessor), sort_keys=True)
        validation = fixture(8, family="A")
        validation["effective_perm_md"] = 1e6
        validation["wag_halfcycle_mo"] = 10
        preprocessor.transform(validation)
        validation.loc[0, "effective_perm_md"] = np.nan
        preprocessor.transform(validation)
        self.assertEqual(snapshot, json.dumps(preprocessing_metadata(preprocessor), sort_keys=True))
        self.assertIn("wag_halfcycle_mo", preprocessing_metadata(preprocessor)["constant_features"])

    def test_raw_median_imputation_precedes_engineering(self):
        X = fixture(8)
        X.loc[[0, 1], "effective_perm_md"] = np.nan
        preprocessor = make_preprocessor().fit(X)
        median = X.effective_perm_md.median()
        self.assertEqual(preprocessing_metadata(preprocessor)["median_imputation"]["effective_perm_md"], median)
        raw = preprocessor.named_steps["raw"].transform(X)
        engineered = preprocessor.named_steps["engineer"].transform(preprocessor.named_steps["imputer"].transform(raw))
        names = FeatureSpec().all_features
        self.assertEqual(engineered[0, names.index("effective_perm_md")], median)
        self.assertAlmostEqual(engineered[0, names.index("log10_perm")], np.log10(median))
        self.assertAlmostEqual(engineered[0, names.index("perm_times_porosity")], median * X.porosity_frac.iloc[0])

    def test_all_missing_training_column_fails_instead_of_silent_drop(self):
        X = fixture()
        X["depth_ft"] = np.nan
        with self.assertRaisesRegex(FeatureValidationError, "median imputation is undefined"):
            make_preprocessor().fit(X)

    def test_invalid_logs_numeric_values_and_denominators_fail(self):
        for name, values in (("effective_perm_md", (0, -1, "bad", np.inf, "nan", True)),
                             ("depth_ft", (0, -1)), ("hcpvi_gas_frac", (-1,)),
                             ("porosity_frac", (0, 1, -1)), ("mmp_full_psi", (0, np.nan))):
            for value in values:
                X = fixture().astype(object)
                X.loc[0, name] = value
                with self.subTest(name=name, value=value), self.assertRaises(FeatureValidationError):
                    make_preprocessor().fit(X)

    def test_context_is_fold_learned_fixed_and_never_refit_at_transform(self):
        X = fixture()
        preprocessor = make_preprocessor().fit(X)
        changed = fixture(4, "A")
        changed["mmp_full_psi"] = 3300
        with self.assertRaisesRegex(FeatureValidationError, "context differs"):
            preprocessor.transform(changed)
        self.assertEqual(preprocessing_metadata(preprocessor)["fixed_context"]["mmp_full_psi"], 3200)
        other = fixture()
        other["mmp_full_psi"] = 3300
        self.assertEqual(preprocessing_metadata(make_preprocessor().fit(other))["fixed_context"]["mmp_full_psi"], 3300)
        X.loc[0, "mmp_full_psi"] = 3300
        with self.assertRaisesRegex(FeatureValidationError, "requires fixed"):
            make_preprocessor().fit(X)

    def test_fit_requires_B_but_inference_can_use_predictor_only_A(self):
        with self.assertRaisesRegex(FeatureValidationError, "reservoir_family=B"):
            make_preprocessor().fit(fixture(family="A"))
        X = fixture()
        preprocessor = make_preprocessor().fit(X)
        subset = fixture(4, "A").loc[:, list(FeatureSpec().raw_features + CONTEXT_COLUMNS)]
        self.assertEqual(preprocessor.transform(subset).shape, (4, 17))

    def test_missing_predictors_duplicate_names_and_arrays_fail(self):
        for X in (fixture().drop(columns="depth_ft"), fixture().to_numpy(),
                  pd.concat([fixture(), fixture()[["depth_ft"]]], axis=1)):
            with self.assertRaises(FeatureValidationError):
                make_preprocessor().fit(X)

    def test_each_CV_fit_has_its_own_imputer_scaler_context_and_constants(self):
        X = fixture(18)
        X.loc[[0, 4, 9], "effective_perm_md"] = np.nan
        X["wag_halfcycle_mo"] = [3] * 12 + [4] * 6
        folds = list(KFold(3).split(X))
        pipeline = make_model_pipeline(DummyRegressor())
        result = cross_validate(pipeline, X, np.arange(len(X)), cv=folds, return_estimator=True,
                                error_score="raise", n_jobs=1)
        self.assertFalse(hasattr(pipeline.named_steps["preprocess"].named_steps["raw"], "context_"))
        medians = []
        for (train, _), fitted in zip(folds, result["estimator"]):
            metadata = preprocessing_metadata(fitted.named_steps["preprocess"])
            expected = make_preprocessor().fit(X.iloc[train])
            self.assertEqual(metadata, preprocessing_metadata(expected))
            median = metadata["median_imputation"]["effective_perm_md"]
            self.assertEqual(median, X.iloc[train].effective_perm_md.median())
            self.assertEqual(metadata["fit_rows"], len(train))
            medians.append(median)
        self.assertGreater(len(set(medians)), 1)
        third = preprocessing_metadata(result["estimator"][2].named_steps["preprocess"])
        self.assertIn("wag_halfcycle_mo", third["constant_features"])

    def test_clone_resets_fit_and_joblib_round_trip_in_fresh_process(self):
        X = fixture()
        preprocessor = make_preprocessor().fit(X)
        fresh = clone(preprocessor)
        with self.assertRaises(NotFittedError):
            fresh.transform(X)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            joblib.dump(preprocessor, directory / "preprocessor.joblib")
            joblib.dump(X, directory / "input.joblib")
            np.save(directory / "expected.npy", preprocessor.transform(X))
            metadata = preprocessing_metadata(preprocessor)
            self.assertEqual(metadata, preprocessing_metadata(joblib.load(directory / "preprocessor.joblib")))
            code = '''
import sys
from pathlib import Path
from wag.runtime import bootstrap_threads
bootstrap_threads()
import joblib, numpy as np
directory = Path(sys.argv[1])
model = joblib.load(directory / "preprocessor.joblib")
X = joblib.load(directory / "input.joblib")
np.testing.assert_array_equal(model.transform(X), np.load(directory / "expected.npy"))
'''
            result = subprocess.run([sys.executable, "-B", "-c", code, temporary], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_inference_unfitted_and_all_constant_training_fail(self):
        with self.assertRaises(NotFittedError):
            make_preprocessor().transform(fixture())
        X = pd.concat([fixture(1)] * 3, ignore_index=True)
        with self.assertRaisesRegex(FeatureValidationError, "All candidate features are constant"):
            make_preprocessor().fit(X)

    def test_feature_import_has_no_IO_workers_or_CUDA(self):
        code = '''
from wag.runtime import bootstrap_threads
bootstrap_threads()
import multiprocessing, os, sys
before = set(os.listdir("."))
import wag.features, wag.models.classical, wag.cpu
assert set(os.listdir(".")) == before
assert not multiprocessing.active_children()
assert "torch" not in sys.modules
'''
        with tempfile.TemporaryDirectory() as temporary:
            import os
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=temporary,
                                    env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

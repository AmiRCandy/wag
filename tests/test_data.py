"""Integrity and leakage-boundary tests use synthetic CSVs, never A scores."""

import copy
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wag.config import Config, PROJECT_ROOT
from wag.data import (BASE_PREDICTORS, DESIGN_COLUMNS, INJECTION_PREDICTORS, NUMERIC_COLUMNS,
                      REQUIRED_COLUMNS, TARGET, DataIntegrityError, check_test_inputs,
                      file_fingerprint, load_development, load_split_manifest, make_split_manifest,
                      save_audit_artifacts, validate_split_manifest)


def records(n=30, family="B"):
    result = []
    for i in range(n):
        row = {column: 1 for column in NUMERIC_COLUMNS}
        row.update(source="OPM_SPE5_simulation", reservoir_family=family,
                   process_type="Miscible", injectant="SPE5 solvent", producer_cell="7,7,3",
                   injector_cell="1,1,1", hcpvi_source="summary", perm_mult=0.5 + i / 100,
                   effective_perm_md=(0.5 + i / 100) * 250, porosity_frac=0.2,
                   porosity_pct=20, depth_ft=5000, init_pressure_psi=2500,
                   rf_waterflood_total_pct=40, rf_wag_total_pct=40 + (i - 10) / 2,
                   rf_incremental_pct=(i - 10) / 2)
        result.append(row)
    return result


def write_csv(path, rows, columns=REQUIRED_COLUMNS):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.b = self.root / "B.csv"
        self.a = self.root / "WAG_simulation_dataset_A_test.csv"
        self.rows = records()
        write_csv(self.b, self.rows)
        write_csv(self.a, records(family="A"))

    def load(self):
        return load_development(self.b, test_path=self.a)

    def test_valid_counts_negatives_constants_and_no_file_edits(self):
        before = file_fingerprint(self.b)
        data = self.load()
        self.assertEqual(len(data.row_ids), 30)
        self.assertEqual(data.audit["target"]["negative_rows"], 10)
        self.assertEqual(data.audit["target"]["zero_rows"], 1)
        self.assertEqual(data.frame[TARGET].tolist(), [row[TARGET] for row in self.rows])
        self.assertEqual(data.audit["constants"]["porosity_frac"], 0.2)
        self.assertEqual(data.audit["excluded_rows"], [])
        self.assertEqual(before, file_fingerprint(self.b))

    def test_missing_test_file_does_not_block_development(self):
        self.a.unlink()
        with patch("wag.data._read_columns", wraps=__import__("wag.data", fromlist=["_read_columns"])._read_columns) as reader:
            data = self.load()
            make_split_manifest(data, Config())
            self.assertTrue(all(Path(call.args[0]) == self.b for call in reader.call_args_list))

    def test_external_path_rejected_before_read(self):
        with patch("wag.data._read_columns") as reader:
            with self.assertRaises(DataIntegrityError):
                load_development(self.a, test_path=self.a)
            with self.assertRaises(DataIntegrityError):
                load_development(self.a)
            reader.assert_not_called()

    def test_symlink_test_alias_rejected(self):
        alias = self.root / "alias.csv"
        alias.symlink_to(self.a)
        with self.assertRaises(DataIntegrityError):
            load_development(alias, test_path=self.a)

    def test_renamed_A_rejected_by_metadata_before_outcomes(self):
        write_csv(self.b, records(family="A"))
        import wag.data as module
        with patch("wag.data._read_columns", wraps=module._read_columns) as reader:
            with self.assertRaisesRegex(DataIntegrityError, "reservoir_family"):
                self.load()
            self.assertEqual(len(reader.call_args_list), 1)
            self.assertEqual(reader.call_args.args[1], ("source", "reservoir_family"))

    def test_mixed_family_or_wrong_source_rejected(self):
        for column, bad in (("reservoir_family", "A"), ("source", "field_observation")):
            rows = copy.deepcopy(self.rows)
            rows[0][column] = bad
            write_csv(self.b, rows)
            with self.subTest(column=column), self.assertRaises(DataIntegrityError):
                self.load()

    def test_invalid_values_fail_without_cleaning(self):
        before_rows = len(self.rows)
        for column in (TARGET, "effective_perm_md", "cum_oil_stb"):
            for value in ("", "not_numeric", "nan", "inf", "-inf"):
                rows = copy.deepcopy(self.rows)
                rows[3][column] = value
                write_csv(self.b, rows)
                before = file_fingerprint(self.b)
                with self.subTest(column=column, value=value), self.assertRaisesRegex(DataIntegrityError, "nonfinite"):
                    self.load()
                self.assertEqual(file_fingerprint(self.b), before)
        self.assertEqual(len(self.rows), before_rows)

    def test_missing_and_duplicate_headers_fail(self):
        write_csv(self.b, self.rows, [column for column in REQUIRED_COLUMNS if column != TARGET])
        with self.assertRaisesRegex(DataIntegrityError, "Missing"):
            self.load()
        write_csv(self.b, self.rows, list(REQUIRED_COLUMNS) + [TARGET])
        with self.assertRaisesRegex(DataIntegrityError, "Duplicate"):
            self.load()

    def test_missing_text_and_blank_records_are_not_silently_dropped(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["producer_cell"] = ""
        write_csv(self.b, rows)
        with self.assertRaisesRegex(DataIntegrityError, "text metadata"):
            self.load()
        write_csv(self.b, self.rows)
        with self.b.open("a") as stream:
            stream.write("\n")
        with self.assertRaises(DataIntegrityError):
            self.load()

    def test_arithmetic_and_physical_checks(self):
        for column in (TARGET, "porosity_pct", "effective_perm_md"):
            rows = copy.deepcopy(self.rows)
            rows[0][column] += 0.3
            write_csv(self.b, rows)
            with self.subTest(column=column), self.assertRaisesRegex(DataIntegrityError, "Arithmetic"):
                self.load()
        rows = copy.deepcopy(self.rows)
        rows[0]["depth_ft"] = -1
        write_csv(self.b, rows)
        with self.assertRaisesRegex(DataIntegrityError, "Physical"):
            self.load()

    def test_full_and_design_duplicates_are_explicit_failures(self):
        rows = self.rows + [copy.deepcopy(self.rows[0])]
        write_csv(self.b, rows)
        with self.assertRaisesRegex(DataIntegrityError, "duplicate simulation"):
            self.load()
        rows[-1][TARGET] += 1
        rows[-1]["rf_wag_total_pct"] += 1
        write_csv(self.b, rows)
        with self.assertRaisesRegex(DataIntegrityError, "Repeated design"):
            self.load()

    def test_row_ids_survive_column_order_and_target_updates(self):
        original = self.load().row_ids
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row[TARGET] += 1
            row["rf_wag_total_pct"] += 1
        write_csv(self.b, rows, tuple(reversed(REQUIRED_COLUMNS)))
        self.assertEqual(self.load().row_ids, original)
        write_csv(self.b, list(reversed(rows)))
        self.assertEqual(self.load().row_ids, tuple(reversed(original)))

    def test_extra_columns_are_reported_not_selected(self):
        for row in self.rows:
            row["future_output"] = ""
        write_csv(self.b, self.rows, list(REQUIRED_COLUMNS) + ["future_output"])
        data = self.load()
        self.assertEqual(data.audit["extra_columns"], ["future_output"])
        self.assertNotIn("future_output", data.audit["row_id_columns"])

    def test_A_predictor_check_never_selects_outcomes(self):
        data = self.load()
        import wag.data as module
        with patch("wag.data._read_columns", wraps=module._read_columns) as reader:
            report = check_test_inputs(data, self.a)
            selected = reader.call_args.args[1]
            self.assertTrue(set(selected).isdisjoint({TARGET, "rf_wag_total_pct", "cum_oil_stb"}))
        self.assertEqual(report["design_overlap"]["matching_A_rows"], 30)
        self.assertEqual(report["predictor_overlap"]["matching_A_rows"], 30)
        self.assertFalse(report["A_outcomes_accessed"])
        rows = records(family="A")
        for row in rows:
            row[TARGET] = "unreadable_label"
            row["rf_wag_total_pct"] = "unreadable_total"
        write_csv(self.a, rows)
        changed = check_test_inputs(data, self.a)
        self.assertEqual(changed["design_overlap"], report["design_overlap"])
        self.assertEqual(changed["predictor_overlap"], report["predictor_overlap"])
        self.assertNotEqual(changed["fingerprint"]["sha256"], report["fingerprint"]["sha256"])

    def test_predictor_only_A_with_no_outcome_columns(self):
        columns = tuple(dict.fromkeys(("source", "reservoir_family") + DESIGN_COLUMNS + BASE_PREDICTORS + INJECTION_PREDICTORS["volumes"]))
        write_csv(self.a, records(family="A"), columns)
        self.assertEqual(check_test_inputs(self.load(), self.a)["rows"], 30)

    def test_holdout_inner_splits_disjoint_reproducible_and_B_only(self):
        data = self.load()
        manifest = make_split_manifest(data, Config())
        self.assertEqual(manifest, make_split_manifest(data, Config()))
        partition = manifest["outer_partitions"][0]
        train = set(partition["development"]["row_ids"])
        valid = set(partition["validation"]["row_ids"])
        self.assertEqual((len(train), len(valid)), (24, 6))
        self.assertFalse(train & valid)
        self.assertEqual(train | valid, set(data.row_ids))
        for fold in partition["inner_folds"]:
            self.assertTrue(set(fold["train"]["row_ids"]) <= train)
            self.assertTrue(set(fold["validation"]["row_ids"]) <= train)
        other = make_split_manifest(data, Config(seed=52))
        self.assertNotEqual(manifest["outer_partitions"], other["outer_partitions"])

    def test_nested_outer_oof_coverage_and_inner_boundaries(self):
        data = self.load()
        manifest = make_split_manifest(data, Config(validation="nested-cv"))
        seen = []
        for outer in manifest["outer_partitions"]:
            seen.extend(outer["validation"]["row_ids"])
            development = set(outer["development"]["row_ids"])
            for fold in outer["inner_folds"]:
                self.assertTrue(set(fold["train"]["row_ids"]) <= development)
                self.assertTrue(set(fold["validation"]["row_ids"]) <= development)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(data.row_ids))
        validate_split_manifest(manifest, data)

    def test_small_data_folds_and_empty_file_fail_clearly(self):
        write_csv(self.b, records(3))
        with self.assertRaisesRegex(DataIntegrityError, "Too few"):
            make_split_manifest(self.load(), Config())
        write_csv(self.b, [])
        with self.assertRaisesRegex(DataIntegrityError, "no records"):
            self.load()

    def test_serialized_split_ids_and_configuration_guard(self):
        data = self.load()
        manifest = make_split_manifest(data, Config())
        path = self.root / "splits.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(load_split_manifest(path, data, Config()), manifest)
        with self.assertRaisesRegex(DataIntegrityError, "configuration"):
            load_split_manifest(path, data, Config(seed=19))

    def test_corrupted_fingerprint_ids_and_overlap_rejected(self):
        data = self.load()
        manifest = make_split_manifest(data, Config())
        corrupt = copy.deepcopy(manifest)
        corrupt["fingerprint"]["sha256"] = "bad"
        with self.assertRaisesRegex(DataIntegrityError, "fingerprint"):
            validate_split_manifest(corrupt, data)
        corrupt = copy.deepcopy(manifest)
        corrupt["outer_partitions"][0]["validation"]["row_ids"][0] = "B:bad"
        with self.assertRaisesRegex(DataIntegrityError, "misaligned"):
            validate_split_manifest(corrupt, data)
        corrupt = copy.deepcopy(manifest)
        train = corrupt["outer_partitions"][0]["development"]
        valid = corrupt["outer_partitions"][0]["validation"]
        valid["positions"][0] = train["positions"][0]
        valid["row_ids"][0] = train["row_ids"][0]
        with self.assertRaisesRegex(DataIntegrityError, "overlap"):
            validate_split_manifest(corrupt, data)
        with self.assertRaisesRegex(DataIntegrityError, "Invalid split"):
            validate_split_manifest({}, data)
        corrupt = copy.deepcopy(manifest)
        del corrupt["outer_partitions"][0]["validation"]["positions"]
        with self.assertRaisesRegex(DataIntegrityError, "Malformed"):
            validate_split_manifest(corrupt, data)

    def test_CSV_changed_during_loading_is_rejected(self):
        original = file_fingerprint(self.b)
        changed = dict(original, sha256="changed")
        with patch("wag.data.file_fingerprint", side_effect=[original, changed]):
            with self.assertRaisesRegex(DataIntegrityError, "changed during loading"):
                self.load()

    def test_A_label_changes_cannot_change_B_splits(self):
        before = make_split_manifest(self.load(), Config())
        rows = records(family="A")
        for row in rows:
            row[TARGET] = "not_a_number"
        write_csv(self.a, rows)
        after = make_split_manifest(self.load(), Config())
        self.assertEqual(before, after)

    def test_artifact_completion_marker_and_no_overwrite(self):
        data = self.load()
        splits = make_split_manifest(data, Config())
        output = self.root / "audit"
        save_audit_artifacts(output, data.audit, splits, {"status": "audit_complete"})
        marker = json.loads((output / "audit_manifest.json").read_text())
        for name, fingerprint in marker["artifact_fingerprints"].items():
            self.assertEqual(file_fingerprint(output / name), fingerprint)
        self.assertEqual(load_split_manifest(output / "split_manifest.json", data, Config()), splits)
        with self.assertRaises(FileExistsError):
            save_audit_artifacts(output, {}, {}, {})

    def test_cli_audit_works_without_A_and_without_gpu(self):
        self.a.unlink()
        output = self.root / "cli_audit"
        args = [sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "audit", "--train-csv", str(self.b),
                "--test-csv", str(self.a), "--run-dir", str(output), "--device", "cuda:999",
                "--workers", "8", "--threads", "1"]
        result = subprocess.run(args, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["B_rows"], 30)
        marker = json.loads((output / "audit_manifest.json").read_text())
        self.assertFalse(marker["A_predictors_accessed"])
        self.assertIsNone(marker["A_fingerprint"])
        self.assertFalse(marker["training_performed"])
        self.assertEqual(marker["runtime"]["thread_limits"]["effective"]["OMP_NUM_THREADS"], "1")
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)

    def test_cli_nested_cv_and_opt_in_A_predictors(self):
        output = self.root / "nested"
        result = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "audit",
                                 "--train-csv", str(self.b), "--test-csv", str(self.a),
                                 "--validation", "nested-cv", "--check-test-inputs", "--run-dir", str(output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["protocol"], "nested-cv")
        self.assertEqual(len(summary["partitions"]), 5)
        self.assertFalse(summary["A_outcomes_accessed"])

    def test_cli_failure_leaves_no_partial_output(self):
        write_csv(self.b, records(family="A"))
        output = self.root / "rejected"
        result = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "main.py"), "audit",
                                 "--train-csv", str(self.b), "--test-csv", str(self.a), "--run-dir", str(output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("reservoir_family", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

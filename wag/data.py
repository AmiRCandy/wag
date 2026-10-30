"""SPE5 CSV integrity, B-only development data and persisted split identities.

Import-safe: pandas/numpy/sklearn are imported only inside explicit operations.
No exclusions, imputation, fitting, or target-based stratification occur here.
"""

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .runtime import task_seed


SCHEMA_VERSION = "spe5-v3-1"
TARGET = "rf_incremental_pct"
TEXT_COLUMNS = ("source", "process_type", "injectant", "reservoir_family",
                "producer_cell", "injector_cell", "hcpvi_source")
NUMERIC_COLUMNS = (
    "perm_mult", "effective_perm_md", "porosity_frac", "porosity_pct",
    "init_pressure_psi", "depth_ft", "gas_rate_mscf", "wat_rate_stb", "prod_orat_stb",
    "wag_halfcycle_mo", "wag_ratio_approx", "hcpvi_target", "wag_ratio_target",
    "waterflood_years", "perm_contrast_exp", "oil_visc_mult", "kv_kh_mult",
    "thickness_mult", "cell_size_ft", "mmp_full_psi", "tl_omega", "temperature_F",
    "grid_nx", "grid_ny", "grid_nz", "solvent_fraction", "primary_years", "wag_years",
    "rf_wag_total_pct", "rf_waterflood_total_pct", "cum_oil_stb", "oip_initial_stb",
    "cum_gas_injected", "cum_solvent_injected_mscf", "cum_water_injected_stb",
    "hcpv_rb", "hcpvi_gas_frac", "pvi_water_frac", "wag_ratio_resvol",
    "pvi_water_waterflood_frac", TARGET)
REQUIRED_COLUMNS = TEXT_COLUMNS + NUMERIC_COLUMNS
# Independent exported design/context, excluding outputs and equivalent copies.
# These identify scenarios; they are NOT an automatic model feature selection.
DESIGN_COLUMNS = (
    "effective_perm_md", "porosity_frac", "init_pressure_psi", "depth_ft",
    "gas_rate_mscf", "wat_rate_stb", "prod_orat_stb", "wag_halfcycle_mo",
    "hcpvi_target", "wag_ratio_target", "waterflood_years", "perm_contrast_exp",
    "oil_visc_mult", "kv_kh_mult", "thickness_mult", "cell_size_ft", "mmp_full_psi",
    "tl_omega", "process_type", "injectant", "temperature_F", "grid_nx", "grid_ny",
    "grid_nz", "producer_cell", "injector_cell", "solvent_fraction", "primary_years", "wag_years")
BASE_PREDICTORS = ("effective_perm_md", "porosity_frac", "init_pressure_psi", "depth_ft",
                   "prod_orat_stb", "wag_halfcycle_mo", "waterflood_years",
                   "perm_contrast_exp", "oil_visc_mult")
INJECTION_PREDICTORS = {"volumes": ("hcpvi_gas_frac", "wag_ratio_resvol"),
                        "design": ("hcpvi_target", "wag_ratio_target"),
                        "rates": ("gas_rate_mscf", "wat_rate_stb")}


class DataIntegrityError(ValueError):
    """Invalid data are rejected in full, never silently cleaned."""


@dataclass(frozen=True)
class DevelopmentData:
    frame: Any
    row_ids: tuple[str, ...]
    fingerprint: dict
    audit: dict


def file_fingerprint(path):
    path = Path(path).resolve()
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def csv_header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        try:
            header = next(csv.reader(stream, strict=True))
        except (StopIteration, csv.Error) as exc:
            raise DataIntegrityError("CSV is empty or has an invalid header") from exc
    if not header or any(not name.strip() for name in header):
        raise DataIntegrityError("CSV column names must be nonempty")
    if len(header) != len(set(header)):
        raise DataIntegrityError("Duplicate CSV column names are not allowed")
    return header


def _require_columns(header, columns):
    missing = sorted(set(columns) - set(header))
    if missing:
        raise DataIntegrityError(f"Missing required CSV columns: {missing}")


def _read_columns(path, columns=None):
    import pandas as pd
    # No NA inference: explicit validation distinguishes missing from zero.
    return pd.read_csv(path, usecols=list(columns) if columns else None,
                       dtype=str, keep_default_na=False, skip_blank_lines=False, encoding="utf-8-sig")


def _assert_family(frame, family):
    if len(frame) == 0:
        raise DataIntegrityError("CSV contains no records")
    for column, expected in (("source", "OPM_SPE5_simulation"), ("reservoir_family", family)):
        if column not in frame or not frame[column].eq(expected).all():
            raise DataIntegrityError(f"Expected {column}={expected!r} in every row; development accepts B only")


def _numeric(frame, columns):
    import numpy as np
    import pandas as pd
    result = frame.copy()
    failures = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = ~np.isfinite(values.to_numpy(dtype=float))
        if invalid.any():
            failures.append({"column": column, "count": int(invalid.sum()),
                             "record_positions": np.flatnonzero(invalid)[:10].tolist()})
        result[column] = values
    if failures:
        raise DataIntegrityError(f"Missing, nonnumeric or nonfinite values (zero-based record positions): {failures}")
    return result


def _check_text(frame, columns):
    for column in columns:
        if frame[column].str.strip().eq("").any():
            raise DataIntegrityError(f"Missing text metadata in {column}")


def _signatures(frame, columns):
    """Full SHA-256 of normalized raw coordinates; never includes outcomes."""
    numeric = set(NUMERIC_COLUMNS)
    signatures = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        values = [float(value).hex() if name in numeric and float(value) != 0 else
                  (float(0).hex() if name in numeric else str(value))
                  for name, value in zip(columns, row)]
        payload = json.dumps([SCHEMA_VERSION, list(columns), values], separators=(",", ":"))
        signatures.append(hashlib.sha256(payload.encode()).hexdigest())
    return signatures


def _check_physical_values(frame):
    import numpy as np
    conditions = {"porosity_frac in (0,1)": frame.porosity_frac.between(0, 1, inclusive="neither")}
    for column in ("perm_mult", "effective_perm_md", "init_pressure_psi", "depth_ft",
                   "gas_rate_mscf", "wat_rate_stb", "prod_orat_stb", "hcpvi_target",
                   "wag_ratio_target", "perm_contrast_exp", "oil_visc_mult", "kv_kh_mult",
                   "thickness_mult", "cell_size_ft", "mmp_full_psi", "oip_initial_stb", "hcpv_rb"):
        conditions[f"{column} > 0"] = frame[column] > 0
    for column in ("wag_halfcycle_mo", "waterflood_years", "grid_nx", "grid_ny", "grid_nz", "wag_years"):
        conditions[f"{column} positive integer"] = (frame[column] > 0) & (frame[column] == np.floor(frame[column]))
    for column in ("cum_oil_stb", "cum_gas_injected", "cum_solvent_injected_mscf",
                   "cum_water_injected_stb", "hcpvi_gas_frac", "pvi_water_frac",
                   "wag_ratio_resvol", "pvi_water_waterflood_frac", "primary_years"):
        conditions[f"{column} >= 0"] = frame[column] >= 0
    for column in ("rf_wag_total_pct", "rf_waterflood_total_pct"):
        conditions[f"{column} in [0,100]"] = frame[column].between(-1e-5, 100 + 1e-5)
    conditions["incremental in [-100,100]"] = frame[TARGET].between(-100 - 1e-5, 100 + 1e-5)
    failures = {rule: int((~valid).sum()) for rule, valid in conditions.items() if not valid.all()}
    if failures:
        raise DataIntegrityError(f"Physical/schema value checks failed: {failures}")


def load_development(path, *, test_path=None):
    """Only B is accepted. A aliases/names are rejected before label parsing.

    Metadata-only preflight rejects a renamed A CSV before its outcomes are read.
    Strict schema version 1 requires all 48 simulation columns; extra columns are
    reported but cannot become predictors automatically. Counts are not hardcoded.
    """
    import numpy as np
    path = Path(path).resolve()
    if test_path is not None and path == Path(test_path).resolve():
        raise DataIntegrityError("The configured external test path cannot be used for B development")
    if "_a_test" in path.name.lower() or path.name == "WAG_field_evaluation_dataset_FULL.csv":
        raise DataIntegrityError("External A/field data cannot be used for B development")
    before = file_fingerprint(path)
    header = csv_header(path)
    _require_columns(header, REQUIRED_COLUMNS)
    _assert_family(_read_columns(path, ("source", "reservoir_family")), "B")
    raw = _read_columns(path)
    _assert_family(raw, "B")
    _check_text(raw, TEXT_COLUMNS)
    frame = _numeric(raw, NUMERIC_COLUMNS)
    _check_physical_values(frame)
    identities = {
        "incremental_equals_wag_minus_waterflood": (frame[TARGET], frame.rf_wag_total_pct - frame.rf_waterflood_total_pct),
        "porosity_pct_equals_fraction_times_100": (frame.porosity_pct, frame.porosity_frac * 100),
        # sim_v3.py preserves mean layer permeability 250 mD before scaling.
        "effective_perm_equals_perm_mult_times_250": (frame.effective_perm_md, frame.perm_mult * 250)}
    errors = {}
    for name, (actual, expected) in identities.items():
        valid = np.isclose(actual, expected, rtol=1e-8, atol=1e-7)
        errors[name] = float(np.max(np.abs(actual - expected)))
        if not valid.all():
            raise DataIntegrityError(f"Arithmetic identity {name} failed in {int((~valid).sum())} rows")
    full_duplicates = int(frame.duplicated(subset=list(REQUIRED_COLUMNS)).sum())
    if full_duplicates:
        raise DataIntegrityError(f"{full_duplicates} duplicate simulation rows; no rows were removed")
    signatures = _signatures(frame, DESIGN_COLUMNS)
    if len(set(signatures)) != len(signatures):
        raise DataIntegrityError("Repeated design coordinates cannot be randomly split as independent scenarios; no rows were removed")
    row_ids = tuple("B:" + value for value in signatures)
    after = file_fingerprint(path)
    if before != after:
        raise DataIntegrityError("CSV changed during loading; retry from a stable snapshot")
    constants = {column: frame[column].iloc[0].item() if hasattr(frame[column].iloc[0], "item") else frame[column].iloc[0]
                 for column in header if frame[column].nunique(dropna=False) == 1}
    audit = {"schema_version": SCHEMA_VERSION, "status": "passed", "role": "development_B",
             "fingerprint": after, "rows": len(frame), "columns": header,
             "extra_columns": sorted(set(header) - set(REQUIRED_COLUMNS)), "constants": constants,
             "missing_or_nonfinite_known_values": 0, "duplicate_rows": 0, "duplicate_designs": 0,
             "target": {"name": TARGET, "units": "percentage_points_of_initial_oil_in_place",
                        "negative_rows": int((frame[TARGET] < 0).sum()), "zero_rows": int((frame[TARGET] == 0).sum()),
                        "min": float(frame[TARGET].min()), "max": float(frame[TARGET].max())},
             "identity_max_absolute_errors": errors, "excluded_rows": [],
             "row_id_policy": "B:sha256(schema_version,ordered_design_columns,normalized_coordinates)",
             "row_id_columns": list(DESIGN_COLUMNS),
             "notes": ["All rows retained; no imputation or feature selection",
                       "SPE5 simulated B, not observed field data",
                       "Integrity does not establish simulator convergence or statistical independence"]}
    return DevelopmentData(frame, row_ids, after, audit)


def check_test_inputs(development, test_path, *, injection_mode="volumes"):
    """Optional A predictor-only diagnostic; outcome columns are never selected.

    Hashes the raw file for provenance, but never parses/returns A outcomes.
    Even a predictor-only A CSV without labels is sufficient for this operation.
    """
    _assert_family(development.frame, "B")
    if injection_mode not in INJECTION_PREDICTORS:
        raise DataIntegrityError("Unknown injection mode for predictor diagnostic")
    predictors = BASE_PREDICTORS + INJECTION_PREDICTORS[injection_mode]
    columns = tuple(dict.fromkeys(("source", "reservoir_family") + DESIGN_COLUMNS + predictors))
    path = Path(test_path).resolve()
    if path == Path(development.fingerprint["path"]):
        raise DataIntegrityError("External A predictor check cannot use the B development path")
    before = file_fingerprint(path)
    _require_columns(csv_header(path), columns)
    frame = _read_columns(path, columns)
    _assert_family(frame, "A")
    _check_text(frame, [name for name in columns if name in TEXT_COLUMNS])
    frame = _numeric(frame, [name for name in columns if name in NUMERIC_COLUMNS])
    after = file_fingerprint(path)
    if before != after:
        raise DataIntegrityError("A CSV changed during predictor inspection")
    def overlap(names):
        b = set(_signatures(development.frame, names))
        a = _signatures(frame, names)
        return {"columns": list(names), "matching_A_rows": sum(value in b for value in a),
                "unique_shared_coordinates": len(b & set(a))}
    return {"role": "external_A_predictor_diagnostic", "rows": len(frame), "fingerprint": after,
            "selected_columns": list(columns), "A_outcomes_accessed": False,
            "design_overlap": overlap(DESIGN_COLUMNS), "predictor_overlap": overlap(predictors),
            "notes": ["Diagnostic only; cannot change split, features or model selection",
                      "Zero exact overlap is not proof of independent designs"]}


def _partition(indices, row_ids):
    positions = [int(value) for value in indices]
    return {"positions": positions, "row_ids": [row_ids[value] for value in positions], "n": len(positions)}


def make_split_manifest(development, config):
    """B-only holdout or nested-CV indices; all future models share this artifact."""
    import numpy as np
    from importlib import metadata
    from sklearn.model_selection import KFold, train_test_split
    _assert_family(development.frame, "B")
    n = len(development.frame)
    if len(development.row_ids) != n or len(set(development.row_ids)) != n:
        raise DataIntegrityError("Development row identities are not unique/aligned")
    positions = np.arange(n)
    def inner(indices, phase, fold):
        if len(indices) < config.inner_folds:
            raise DataIntegrityError("Too few development records for the configured inner folds")
        seed = task_seed(config.seed, "split", phase, fold)
        return [{"fold": number, "seed": seed,
                 "train": _partition(indices[train], development.row_ids),
                 "validation": _partition(indices[valid], development.row_ids)}
                for number, (train, valid) in enumerate(KFold(config.inner_folds, shuffle=True, random_state=seed).split(indices))]
    if config.validation == "holdout":
        train, valid = train_test_split(positions, test_size=config.validation_fraction,
                                       random_state=config.seed, shuffle=True)
        outer = [{"fold": 0, "development": _partition(train, development.row_ids),
                  "validation": _partition(valid, development.row_ids),
                  "inner_folds": inner(train, "holdout_inner", 0)}]
    else:
        if n < config.outer_folds:
            raise DataIntegrityError("Too few records for the configured outer folds")
        outer = [{"fold": fold, "development": _partition(train, development.row_ids),
                  "validation": _partition(valid, development.row_ids),
                  "inner_folds": inner(train, "nested_inner", fold)}
                 for fold, (train, valid) in enumerate(KFold(config.outer_folds, shuffle=True, random_state=config.seed).split(positions))]
    manifest = {"schema_version": 1, "role": "development_B", "fingerprint": development.fingerprint,
                "protocol": config.validation, "seed": config.seed,
                "validation_fraction": config.validation_fraction, "inner_folds": config.inner_folds,
                "outer_folds": config.outer_folds if config.validation == "nested-cv" else 1,
                "row_id_policy": development.audit["row_id_policy"],
                "rows": [{"position": i, "csv_record_number": i + 1, "row_id": value}
                         for i, value in enumerate(development.row_ids)],
                "outer_partitions": outer, "all_B_inner_folds": inner(positions, "all_B_inner", 0),
                "interpretation": "B model-selection validation; A is absent from all splits",
                "split_libraries": {"numpy": np.__version__, "scikit-learn": metadata.version("scikit-learn")}}
    validate_split_manifest(manifest, development)
    return manifest


def validate_split_manifest(manifest, development):
    """Reject data mismatch, wrong identities, overlapping and incomplete partitions."""
    try:
        _validate_split_manifest(manifest, development)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise DataIntegrityError(f"Malformed split manifest: {exc}") from exc


def _validate_split_manifest(manifest, development):
    _assert_family(development.frame, "B")
    expected_ids = tuple("B:" + value for value in _signatures(development.frame, DESIGN_COLUMNS))
    if development.row_ids != expected_ids:
        raise DataIntegrityError("Loaded B row identities are misaligned with design coordinates")
    if manifest.get("schema_version") != 1 or manifest.get("role") != "development_B":
        raise DataIntegrityError("Invalid split schema/role")
    if manifest.get("fingerprint") != development.fingerprint:
        raise DataIntegrityError("Split fingerprint does not match the loaded B data")
    expected_rows = [{"position": i, "csv_record_number": i + 1, "row_id": value}
                     for i, value in enumerate(development.row_ids)]
    if manifest.get("rows") != expected_rows:
        raise DataIntegrityError("Split row identities/order do not match B")
    if type(manifest["inner_folds"]) is not int or manifest["inner_folds"] < 2:
        raise DataIntegrityError("Invalid inner-fold count")
    if type(manifest["outer_folds"]) is not int or manifest["outer_folds"] < 1:
        raise DataIntegrityError("Invalid outer-fold count")
    universe = set(range(len(development.row_ids)))
    def partition(value):
        positions = value["positions"]
        if not positions or any(type(i) is not int or i not in universe for i in positions):
            raise DataIntegrityError("Invalid/empty partition positions")
        if len(positions) != len(set(positions)) or value != _partition(positions, development.row_ids):
            raise DataIntegrityError("Partition row IDs/counts are misaligned or repeated")
        return set(positions)
    def pair(train, valid, parent):
        train, valid = partition(train), partition(valid)
        if train & valid or train | valid != parent:
            raise DataIntegrityError("Partitions overlap or do not cover their parent dataset")
        return train, valid
    def inner(folds, parent):
        if len(folds) != manifest["inner_folds"]:
            raise DataIntegrityError("Incorrect inner-fold count")
        seen = []
        for number, fold in enumerate(folds):
            if fold["fold"] != number:
                raise DataIntegrityError("Invalid inner-fold numbering")
            _, valid = pair(fold["train"], fold["validation"], parent)
            seen.extend(valid)
        if len(seen) != len(set(seen)) or set(seen) != parent:
            raise DataIntegrityError("Inner validation does not cover each parent row exactly once")
    outer = manifest["outer_partitions"]
    if manifest["protocol"] not in ("holdout", "nested-cv") or len(outer) != manifest["outer_folds"]:
        raise DataIntegrityError("Incorrect outer protocol/fold count")
    if manifest["protocol"] == "holdout" and len(outer) != 1:
        raise DataIntegrityError("Holdout requires exactly one outer partition")
    seen = []
    for number, fold in enumerate(outer):
        if fold["fold"] != number:
            raise DataIntegrityError("Invalid outer-fold numbering")
        train, valid = pair(fold["development"], fold["validation"], universe)
        inner(fold["inner_folds"], train)
        seen.extend(valid)
    if manifest["protocol"] == "nested-cv" and (len(seen) != len(set(seen)) or set(seen) != universe):
        raise DataIntegrityError("Outer validation does not cover each B row exactly once")
    inner(manifest["all_B_inner_folds"], universe)


def load_split_manifest(path, development, config):
    """Read saved IDs only when the B fingerprint and declared protocol match."""
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_split_manifest(manifest, development)
    expected = {"protocol": config.validation, "seed": config.seed,
                "validation_fraction": config.validation_fraction,
                "inner_folds": config.inner_folds,
                "outer_folds": config.outer_folds if config.validation == "nested-cv" else 1}
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise DataIntegrityError("Saved split configuration does not match requested protocol/seed/folds")
    return manifest


def save_audit_artifacts(run_dir, audit, splits, manifest):
    """Fresh-directory, per-file atomic writes; manifest is the completion marker."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(manifest)
    manifest["artifact_fingerprints"] = {}
    for name, value in (("data_audit.json", audit), ("split_manifest.json", splits),
                        ("audit_manifest.json", manifest)):
        target = directory / name
        temporary = directory / ("." + name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(target)
        if name != "audit_manifest.json":
            manifest["artifact_fingerprints"][name] = file_fingerprint(target)
    return directory

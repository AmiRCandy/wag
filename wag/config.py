"""Standard-library-only configuration; no data loading or numeric imports."""

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import re
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    train_csv: str = "data/WAG_simulation_dataset_B_train.csv"
    test_csv: str = "data/WAG_simulation_dataset_A_test.csv"
    run_dir: str | None = None
    classical_config: str = "configs/classical.json"
    neural_config: str = "configs/neural.json"
    model_scope: str = "all"
    resume: bool = False
    workers: int = 8
    threads: int = 1
    device: str = "cuda:0"
    seed: int = 42
    validation: str = "holdout"
    validation_fraction: float = 0.2
    inner_folds: int = 3
    outer_folds: int = 5
    search: str = "randomized"
    search_candidates: int = 20
    injection_mode: str = "volumes"

    def __post_init__(self):
        for key in ("workers", "threads", "seed", "inner_folds", "outer_folds", "search_candidates"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if not 1 <= self.workers <= 8:
            raise ValueError("workers must be between 1 and 8")
        if self.threads != 1:
            raise ValueError("threads must be exactly 1")
        if type(self.resume) is not bool or (self.resume and self.run_dir is None):
            raise ValueError("resume must be boolean and requires an explicit run_dir")
        if self.model_scope not in ("all", "classical", "neural"):
            raise ValueError("model_scope must be all, classical or neural")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.inner_folds < 2 or self.outer_folds < 2 or self.search_candidates < 1:
            raise ValueError("fold counts must be >=2 and search_candidates must be >=1")
        if type(self.validation_fraction) not in (int, float) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        for key, choices in (("validation", ("holdout", "nested-cv")),
                             ("search", ("randomized", "grid")),
                             ("injection_mode", ("volumes", "design", "rates"))):
            if getattr(self, key) not in choices:
                raise ValueError(f"{key} must be one of {choices}")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", self.device):
            raise ValueError("device must be cpu or an explicit CUDA index such as cuda:0")
        for key in ("train_csv", "test_csv", "run_dir", "classical_config", "neural_config"):
            value = getattr(self, key)
            if value is None and key == "run_dir":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty path string")

    def resolved(self, root=PROJECT_ROOT):
        """Resolve paths without checking existence or opening either dataset."""
        result = asdict(self)
        for key in ("train_csv", "test_csv", "run_dir", "classical_config", "neural_config"):
            if result[key] is not None:
                path = Path(result[key]).expanduser()
                result[key] = str((root / path).resolve())
        return Config(**result)


def load_config(cli_values=None, config_path=None, *, root=PROJECT_ROOT, environ=None):
    """Precedence: CLI > explicit JSON > legacy environment > defaults.

    Only WAG_N_JOBS and WAG_SIM_CSV are mapped. Conflicts are reported.
    Unrecognized settings are errors, not silently ignored configuration.
    """
    environ = os.environ if environ is None else environ
    values = {}
    for old, new in (("WAG_N_JOBS", "workers"), ("WAG_SIM_CSV", "train_csv")):
        if old in environ:
            warnings.warn(f"Legacy {old} maps to {new}; prefer CLI/JSON configuration", UserWarning)
            try:
                values[new] = int(environ[old]) if new == "workers" else environ[old]
            except ValueError as exc:
                raise ValueError(f"{old} must be an integer") from exc
    explicit = {}
    if config_path:
        path = Path(config_path).expanduser()
        with (root / path).open(encoding="utf-8") as stream:
            explicit = json.load(stream)
        if not isinstance(explicit, dict):
            raise ValueError("configuration JSON must be an object")
    valid = {field.name for field in fields(Config)}
    cli_values = {key: value for key, value in (cli_values or {}).items() if value is not None}
    unknown = (explicit.keys() | cli_values.keys()) - valid
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    for layer in (explicit, cli_values):
        for key in values.keys() & layer.keys():
            if values[key] != layer[key]:
                warnings.warn(f"Configuration override {key}: {values[key]!r} -> {layer[key]!r}", UserWarning)
        values.update(layer)
    return Config(**values).resolved(root)

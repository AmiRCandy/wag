"""Explicit runtime setup. Numeric libraries are imported only inside functions."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
              "VECLIB_MAXIMUM_THREADS")
_torch_configured = False
_cpu_pool_active = ContextVar("wag_cpu_pool_active", default=False)
_neural_fit_active = ContextVar("wag_neural_fit_active", default=False)
MINIMUM_VERSIONS = {"numpy": (1, 24), "pandas": (1, 5), "scipy": (1, 10),
                    "scikit-learn": (1, 3), "joblib": (1, 3), "threadpoolctl": (3, 1),
                    "torch": (2, 0), "xgboost": (1, 7), "lightgbm": (3, 3),
                    "matplotlib": (3, 7), "seaborn": (0, 12), "tqdm": (4, 66)}


def bootstrap_threads(threads=1):
    if type(threads) is not int or threads != 1:
        raise ValueError("threads must be exactly 1")
    loaded = sorted(set(sys.modules) & {"numpy", "scipy", "sklearn", "torch", "numexpr"})
    if loaded:
        raise RuntimeError(f"Runtime bootstrap must precede numeric imports: {loaded}. Start a fresh process.")
    previous = {key: os.environ.get(key) for key in THREAD_ENV}
    for key in THREAD_ENV:
        os.environ[key] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    return {"previous": previous, "effective": {key: os.environ[key] for key in THREAD_ENV},
            "OMP_DYNAMIC": "FALSE"}


def task_seed(base_seed, model_id, phase, fold=0, repeat=0):
    payload = json.dumps([base_seed, model_id, phase, fold, repeat], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def dependency_preflight(requirements):
    """Check distributions without importing them or installing anything."""
    versions, missing, outdated = {}, [], []
    for package in requirements:
        try:
            versions[package] = metadata.version(package)
            match = re.match(r"(\d+)\.(\d+)", versions[package])
            minimum = MINIMUM_VERSIONS.get(package)
            if minimum and (not match or tuple(map(int, match.groups())) < minimum):
                outdated.append(f"{package} {versions[package]} (requires >= {'.'.join(map(str, minimum))})")
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError("Missing dependencies: " + ", ".join(missing)
                           + ". Install explicitly in your existing environment; see SETUP.md. Nothing was installed.")
    if outdated:
        raise RuntimeError("Unsupported dependency versions: " + ", ".join(outdated)
                           + ". Review SETUP.md before changing the environment; nothing was upgraded.")
    return versions


def configure_torch(device):
    """Coordinator only; never called at import or from CPU fitting workers."""
    global _torch_configured
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", device):
        raise ValueError("device must be cpu or cuda:N")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("Call bootstrap_threads before importing/configuring PyTorch")
    torch = importlib.import_module("torch")
    if not _torch_configured:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError("Configure PyTorch before tensor work in a fresh process") from exc
        _torch_configured = True
    if device == "cpu":
        return {"requested": device, "selected": "cpu", "name": "CPU",
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads()}
    index = int(device.split(":")[1])
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable. Check your installed PyTorch/CUDA stack or explicitly use --device cpu.")
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(f"Requested {device}, but only {count} CUDA device(s) are visible. Check CUDA_VISIBLE_DEVICES.")
    torch.cuda.set_device(index)
    return {"requested": device, "selected": device, "logical_index": index,
            "name": torch.cuda.get_device_name(index), "visible_count": count,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


@contextmanager
def cpu_parallelism(workers=8, threads=1):
    """One active pool; callers must use single-threaded estimators and bounded pre_dispatch."""
    if type(workers) is not int or not 1 <= workers <= 8 or type(threads) is not int or threads != 1:
        raise ValueError("CPU policy requires 1..8 workers and exactly 1 thread")
    if any(os.environ.get(key) != "1" for key in THREAD_ENV):
        raise RuntimeError("Call bootstrap_threads before creating a CPU fitting pool")
    import multiprocessing
    if _cpu_pool_active.get() or _neural_fit_active.get() or multiprocessing.current_process().name != "MainProcess":
        raise RuntimeError("Nested fitting pools are forbidden; CPU workers fit sequentially")
    from joblib import parallel_config
    token = _cpu_pool_active.set(True)
    try:
        with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1,
                             max_nbytes="1M", mmap_mode="r"):
            yield {"n_jobs": workers, "pre_dispatch": workers}
    finally:
        _cpu_pool_active.reset(token)


@contextmanager
def neural_execution(device):
    """One neural operation in the main coordinator, never inside a CPU pool."""
    import multiprocessing
    import threading
    if (_cpu_pool_active.get() or _neural_fit_active.get()
            or multiprocessing.current_process().name != "MainProcess"
            or threading.current_thread() is not threading.main_thread()):
        raise RuntimeError("Neural execution must be sequential in the main coordinator, outside CPU pools")
    dependency_preflight(["torch"])
    report = configure_torch(device)  # explicit CUDA errors, never CPU fallback
    if report["torch_threads"] != 1 or report["torch_interop_threads"] != 1:
        raise RuntimeError("PyTorch thread settings changed; neural execution requires exactly one intra/inter-op thread")
    token = _neural_fit_active.set(True)
    try:
        yield report
    finally:
        _neural_fit_active.reset(token)


def git_identity(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        revision = git("rev-parse", "HEAD")
        state = git("status", "--porcelain")
        return {"revision": revision, "dirty": bool(state) if state is not None else None}
    except (OSError, subprocess.TimeoutExpired):
        return {"revision": None, "dirty": None}


def runtime_manifest(config, command, limits, versions, device, root):
    from dataclasses import asdict
    return {"schema_version": 1, "status": "runtime_checked_only", "command": command,
            "config": asdict(config), "runtime": {"python": sys.version,
            "platform": platform.platform(), "executable": sys.executable,
            "packages": versions, "thread_limits": limits, "device": device,
            "cpu_backend": "loky", "max_fitting_workers": config.workers,
            "pre_dispatch": config.workers, "neural_fits": "sequential_coordinator",
            "seed_policy": "sha256(base_seed,model_id,phase,fold,repeat)",
            "determinism": "Task seeds reproducible; byte-identical hardware results not guaranteed"},
            "git": git_identity(root), "data_accessed": False,
            "training_performed": False}


def save_runtime_manifest(manifest, run_dir):
    """An explicit check creates a fresh directory; never overwrite old results."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "runtime_manifest.json"
    temporary = directory / ".runtime_manifest.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(target)
    return target

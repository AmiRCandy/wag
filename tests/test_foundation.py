"""Part 1 safeguards; tests never load reservoir datasets or fit models."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from wag.config import Config, PROJECT_ROOT, load_config
from wag import runtime


def invoke(*args, cwd=PROJECT_ROOT, env=None):
    return subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "main.py"), *args],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=30)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_match_json(self):
        with (PROJECT_ROOT / "configs/default.json").open() as stream:
            self.assertEqual(json.load(stream), asdict(Config()))

    def test_resource_and_device_validation(self):
        for value in (-1, 0, 9, 64, True, 8.0, "8"):
            with self.subTest(workers=value), self.assertRaises(ValueError):
                Config(workers=value)
        for value in (0, 2, True, "1"):
            with self.subTest(threads=value), self.assertRaises(ValueError):
                Config(threads=value)
        for value in ("cuda", "cuda:-1", "cuda:01", "gpu:0", "cpu:0", None):
            with self.subTest(device=value), self.assertRaises(ValueError):
                Config(device=value)
        self.assertEqual(Config(device="cuda:3").device, "cuda:3")

    def test_precedence_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / "settings.json"
            config_file.write_text(json.dumps({"workers": 4, "seed": 51}))
            with self.assertWarns(UserWarning):
                config = load_config({"workers": 2}, "settings.json", root=root,
                                     environ={"WAG_N_JOBS": "8"})
            self.assertEqual(config.workers, 2)
            self.assertEqual(config.seed, 51)
            self.assertEqual(config.train_csv, str(root / Config().train_csv))
            self.assertFalse(Path(config.train_csv).exists())

    def test_unknown_and_malformed_settings(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            load_config({"unexpected": True}, environ={})
        with self.assertRaisesRegex(ValueError, "WAG_N_JOBS"):
            load_config(environ={"WAG_N_JOBS": "many"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "object"):
                load_config(config_path=str(path), environ={})


class RuntimeTests(unittest.TestCase):
    def test_bootstrap_overrides_limits_and_keeps_visibility(self):
        code = '''
import os
from wag.runtime import bootstrap_threads, THREAD_ENV
report = bootstrap_threads()
assert report["previous"]["OPENBLAS_NUM_THREADS"] == "32"
assert all(os.environ[key] == "1" for key in THREAD_ENV)
assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                                env={**os.environ, "OPENBLAS_NUM_THREADS": "32", "CUDA_VISIBLE_DEVICES": "3"},
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bootstrap_refuses_late_import(self):
        with patch.dict(sys.modules, {"numpy": MagicMock()}):
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                runtime.bootstrap_threads()

    def test_seed_order_and_process_stability(self):
        first = runtime.task_seed(42, "MLP", "inner", 2, 0)
        self.assertNotEqual(first, runtime.task_seed(42, "MLP", "inner", 3, 0))
        runtime.task_seed(42, "Ridge", "refit")
        self.assertEqual(first, runtime.task_seed(42, "MLP", "inner", 2, 0))
        command = "from wag.runtime import task_seed; print(task_seed(42,'MLP','inner',2,0))"
        for randomized_hash in ("1", "56"):
            result = subprocess.run([sys.executable, "-B", "-c", command], cwd=PROJECT_ROOT,
                                    env={**os.environ, "PYTHONHASHSEED": randomized_hash},
                                    capture_output=True, text=True, check=True, timeout=10)
            self.assertEqual(int(result.stdout), first)

    def test_missing_dependency_actionable(self):
        with patch.object(runtime.metadata, "version", side_effect=runtime.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Nothing was installed"):
                runtime.dependency_preflight(["torch"])

    def test_gpu_selection_no_fallback_and_once_only_threads(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 2
        torch.cuda.get_device_name.return_value = "Fake GPU 1"
        torch.get_num_threads.return_value = 1
        torch.get_num_interop_threads.return_value = 1
        with patch.dict(os.environ, {key: "1" for key in runtime.THREAD_ENV}), patch.object(runtime, "_torch_configured", False), patch.object(runtime.importlib, "import_module", return_value=torch):
            result = runtime.configure_torch("cuda:1")
            self.assertEqual(result["logical_index"], 1)
            torch.cuda.set_device.assert_called_once_with(1)
            runtime.configure_torch("cpu")
            torch.set_num_interop_threads.assert_called_once_with(1)
            with self.assertRaisesRegex(RuntimeError, "only 2"):
                runtime.configure_torch("cuda:2")
            torch.cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                runtime.configure_torch("cuda:0")

    def test_cpu_never_probes_cuda(self):
        torch = MagicMock()
        with patch.dict(os.environ, {key: "1" for key in runtime.THREAD_ENV}), patch.object(runtime, "_torch_configured", False), patch.object(runtime.importlib, "import_module", return_value=torch):
            self.assertEqual(runtime.configure_torch("cpu")["selected"], "cpu")
            torch.cuda.is_available.assert_not_called()
            torch.cuda.set_device.assert_not_called()

    def test_direct_invalid_device_does_not_import_torch(self):
        with patch.object(runtime.importlib, "import_module") as imported:
            with self.assertRaises(ValueError):
                runtime.configure_torch("cuda")
            imported.assert_not_called()

    def test_outdated_dependency_rejected(self):
        with patch.object(runtime.metadata, "version", return_value="1.2.0"):
            with self.assertRaisesRegex(RuntimeError, "requires >= 1.3"):
                runtime.dependency_preflight(["joblib"])

    def test_real_worker_thread_limits(self):
        code = '''
from wag.runtime import bootstrap_threads, cpu_parallelism, THREAD_ENV
bootstrap_threads()
from joblib import Parallel, delayed
def worker():
    import os, numpy
    from threadpoolctl import threadpool_info
    numpy.dot(numpy.ones((8,8)), numpy.ones((8,8)))
    return os.getpid(), {k:os.environ[k] for k in THREAD_ENV}, threadpool_info()
with cpu_parallelism(2) as options:
    records = Parallel(**options)(delayed(worker)() for _ in range(4))
assert 1 <= len({r[0] for r in records}) <= 2
for _, env, pools in records:
    assert all(v == "1" for v in env.values())
    assert all(p["num_threads"] == 1 for p in pools)
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_cuda_when_available(self):
        try:
            runtime.metadata.version("torch")
        except runtime.metadata.PackageNotFoundError:
            self.skipTest("PyTorch not installed; real GPU validation unavailable")
        code = '''
from wag.runtime import bootstrap_threads, configure_torch
bootstrap_threads()
import torch
if not torch.cuda.is_available():
    raise SystemExit(77)
report = configure_torch("cuda:0")
assert torch.cuda.current_device() == 0
assert report["torch_threads"] == report["torch_interop_threads"] == 1
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 77:
            self.skipTest("CUDA unavailable; real GPU validation skipped")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_manifest_records_overrides_no_data_access(self):
        config = Config(workers=4, device="cpu", seed=123).resolved()
        manifest = runtime.runtime_manifest(config, "train", {}, {}, {"selected": "cpu"}, PROJECT_ROOT)
        self.assertEqual(manifest["config"]["workers"], 4)
        self.assertEqual(manifest["config"]["seed"], 123)
        self.assertFalse(manifest["data_accessed"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "check"
            path = runtime.save_runtime_manifest(manifest, output)
            self.assertEqual(json.loads(path.read_text()), manifest)
            with self.assertRaises(FileExistsError):
                runtime.save_runtime_manifest(manifest, output)

    def test_real_joblib_policy_without_starting_workers(self):
        code = '''
from wag.runtime import bootstrap_threads, cpu_parallelism
bootstrap_threads()
from joblib.parallel import get_active_backend
with cpu_parallelism(8) as options:
    backend, jobs = get_active_backend()
    assert jobs == 8
    assert backend.inner_max_num_threads == 1
    assert options == {"n_jobs": 8, "pre_dispatch": 8}
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        for workers in (-1, 9):
            with self.assertRaises(ValueError), runtime.cpu_parallelism(workers):
                pass


class CLITests(unittest.TestCase):
    def test_help_every_command_from_other_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            for command in (None, "run", "train", "refit", "evaluate", "audit", "report", "plots"):
                args = [command, "--help"] if command else ["--help"]
                result = invoke(*args, cwd=temporary)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_full_dispatch_preflight_and_validation_first(self):
        result = invoke("run", "--no-plots", "--model-scope", "classical", "--classical-config", "configs/not_present_for_test.json", "--device", "cpu")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not_present_for_test.json", result.stderr)
        for args, message in ((["--workers", "9"], "workers"), (["--threads", "2"], "threads"),
                              (["--device", "cuda"], "device")):
            result = invoke("train", *args)
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)

    def test_imports_have_no_numeric_imports_writes_processes_or_cuda(self):
        code = '''
import pathlib, subprocess, sys
from unittest.mock import patch
def forbidden(*args, **kwargs):
    raise AssertionError("Side effect during import")
with patch("subprocess.Popen", forbidden), patch("pathlib.Path.open", forbidden), patch("pathlib.Path.mkdir", forbidden):
    import main, wag, wag.config, wag.runtime, wag.cli, wag.models, wag.data
assert not set(sys.modules) & {"numpy", "sklearn", "torch", "joblib"}
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=temporary,
                                    env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_runtime_check_stdout_and_saved_config(self):
        result = invoke("audit", "--check-runtime", "--workers", "3", "--seed", "9")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["config"]["workers"], 3)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest"
            result = invoke("audit", "--check-runtime", "--run-dir", str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "runtime_manifest.json").read_text())
            self.assertEqual(manifest["config"]["run_dir"], str(output))
            self.assertFalse(manifest["training_performed"])
            result = invoke("audit", "--check-runtime", "--run-dir", str(output))
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()

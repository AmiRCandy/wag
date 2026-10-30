"""Sequential neural fit utilities; no PyTorch import until explicit fitting/loading.

Part 6 will provide experiment orchestration; Part 7 will write frozen artifacts.
This module neither opens datasets nor writes files. Inputs are raw B partitions.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import traceback

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .progress import progress
from .config import PROJECT_ROOT
from .features import FeatureSpec, make_preprocessor, preprocessing_metadata
from .models.classical import training_targets
from .runtime import neural_execution, task_seed


MODEL_IDS = ("NN_MLP", "NN_PINN_HARD")


def validate_neural_settings(settings):
    required = {"version", "enabled_models", "hidden_layers", "max_epochs", "learning_rate",
                "weight_decay", "early_stopping", "bounds", "constraints"}
    if not isinstance(settings, dict) or settings.keys() != required or type(settings["version"]) is not int or settings["version"] != 1:
        raise ValueError("Neural settings require the exact version-1 schema")
    enabled = settings["enabled_models"]
    if not isinstance(enabled, list) or not enabled or any(x not in MODEL_IDS for x in enabled) or len(set(enabled)) != len(enabled):
        raise ValueError("enabled_models must uniquely select NN_MLP / NN_PINN_HARD")
    hidden = settings["hidden_layers"]
    if not isinstance(hidden, list) or not hidden or any(type(x) is not int or x < 1 for x in hidden):
        raise ValueError("hidden_layers must be positive integer widths")
    if type(settings["max_epochs"]) is not int or settings["max_epochs"] < 1:
        raise ValueError("max_epochs must be a positive integer")
    def numeric(value, name, *, positive=False):
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    numeric(settings["learning_rate"], "learning_rate", positive=True)
    numeric(settings["weight_decay"], "weight_decay")
    early = settings["early_stopping"]
    if not isinstance(early, dict) or early.keys() != {"validation_fraction", "patience", "min_delta"}:
        raise ValueError("Invalid early_stopping schema")
    numeric(early["validation_fraction"], "validation_fraction", positive=True)
    if early["validation_fraction"] >= 1 or type(early["patience"]) is not int or early["patience"] < 1:
        raise ValueError("Early split must be in (0,1), patience must be a positive integer")
    numeric(early["min_delta"], "min_delta")
    bounds = settings["bounds"]
    if not isinstance(bounds, list) or len(bounds) != 2 or any(type(x) not in (int, float) or not np.isfinite(x) for x in bounds) or not -100 <= bounds[0] < bounds[1] <= 100:
        raise ValueError("bounds must be ordered finite values inside [-100, 100]")
    constraints = settings["constraints"]
    names = {"gas_monotonicity_weight", "gas_concavity_weight", "real_weight", "collocation_weight", "collocation_count"}
    if not isinstance(constraints, dict) or constraints.keys() != names:
        raise ValueError("Invalid constraints schema; permeability monotonicity is not supported")
    count = constraints["collocation_count"]
    if type(count) is not int or count < 0:
        raise ValueError("collocation_count must be a nonnegative integer")
    for name in names - {"collocation_count"}:
        numeric(constraints[name], name)
    return deepcopy(settings)


def load_neural_settings(path=None):
    path = PROJECT_ROOT / Path(path or "configs/neural.json")
    with path.open(encoding="utf-8") as stream:
        return validate_neural_settings(json.load(stream))


def inner_partition(n, settings, seed, supplied=None):
    """Relative positions in this B development partition, never outer holdout."""
    if n < 3:
        raise ValueError("Neural inner training needs at least three B records")
    train, valid = train_test_split(np.arange(n), test_size=settings["early_stopping"]["validation_fraction"],
                                   random_state=seed) if supplied is None else supplied
    result = []
    for values in (train, valid):
        array = np.asarray(values)
        if array.ndim != 1 or array.dtype.kind not in "iu" or not len(array) or array.min() < 0 or array.max() >= n or len(set(array.tolist())) != len(array):
            raise ValueError("Inner early-stopping positions must be unique, valid and nonempty")
        result.append(array)
    if set(result[0]) & set(result[1]) or set(result[0]) | set(result[1]) != set(range(n)):
        raise ValueError("Inner early-stopping partitions overlap or do not cover B development")
    if len(result[0]) < 2:
        raise ValueError("Neural inner training needs at least two training records")
    return tuple(result)


def imputed_raw(preprocessor, X):
    """Validate fixed context/domains before returning canonical, imputed inputs."""
    selected = preprocessor.named_steps["raw"].transform(X)
    return np.asarray(preprocessor.named_steps["imputer"].transform(selected), dtype=float)[:, :-1]


def collocation_raw(raw_training, count, seed):
    """Bootstrap covariates together, vary only gas inside observed training range.

    No independent engineering-space sampling, extrapolation or synthetic labels.
    Other canonical covariates preserve each sampled real row's dependencies.
    """
    if type(count) is not int or count < 0:
        raise ValueError("collocation count must be nonnegative")
    raw = np.asarray(raw_training, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 11 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Collocation requires finite canonical imputed raw inputs")
    rng = np.random.default_rng(seed)
    result = raw[rng.integers(len(raw), size=count)].copy()
    result[:, -2] = rng.uniform(raw[:, -2].min(), raw[:, -2].max(), count)
    return result


@dataclass
class NeuralResult:
    model_id: str
    status: str
    estimator: object = None
    details: dict = field(default_factory=dict)

    def record(self):
        return {"model_id": self.model_id, "status": self.status, **deepcopy(self.details)}


class NeuralPredictor:
    """Frozen preprocessor, model and differentiable map; predict never fits."""

    def __init__(self, model_id, model, mapping, preprocessor, settings, device="cpu"):
        self.model_id, self.model, self.mapping = model_id, model, mapping
        self.preprocessor, self.settings, self.device = preprocessor, deepcopy(settings), device

    def predict(self, X):
        import torch
        raw = imputed_raw(self.preprocessor, X)
        self.model.eval()
        with threadpool_limits(limits=1), torch.no_grad():
            values = torch.tensor(raw, dtype=torch.float32, device=self.device)
            coordinates = (values - self.mapping.raw_mean) / self.mapping.raw_scale
            prediction = self.model(self.mapping(coordinates))
            if not torch.isfinite(prediction).all():
                raise ValueError("Neural prediction produced nonfinite values")
            return prediction.detach().cpu().numpy()

    def state_bundle(self):
        """In-memory portable state; trusted joblib preprocessor saved separately in Part 7."""
        from .models.neural import cpu_state_dict
        metadata = preprocessing_metadata(self.preprocessor)
        return {"version": 1, "model_id": self.model_id, "settings": deepcopy(self.settings),
                "feature_metadata": metadata, "model_state": cpu_state_dict(self.model),
                "mapping_state": cpu_state_dict(self.mapping)}


def _architecture(model_id, preprocessor, settings):
    from .models.neural import MLP, HardPINN
    active = preprocessing_metadata(preprocessor)["active_features"]
    hidden = tuple(settings["hidden_layers"])
    if model_id == "NN_MLP":
        return MLP(len(active), hidden)
    gas = FeatureSpec(preprocessor.named_steps["raw"].injection_mode).raw_features[-2]
    return HardPINN(len(active), active.index(gas) if gas in active else None, hidden, tuple(settings["bounds"]))


def load_neural_bundle(bundle, preprocessor, *, device="cpu"):
    """Load already-deserialized trusted state on CPU by default; never refit."""
    required = {"version", "model_id", "settings", "feature_metadata", "model_state", "mapping_state"}
    if not isinstance(bundle, dict) or bundle.keys() != required or type(bundle["version"]) is not int or bundle["version"] != 1 or bundle["model_id"] not in MODEL_IDS:
        raise ValueError("Unsupported neural state bundle")
    settings = validate_neural_settings(bundle["settings"])
    if bundle["model_id"] not in settings["enabled_models"]:
        raise ValueError("Saved neural model is not enabled in saved settings")
    if bundle["feature_metadata"] != preprocessing_metadata(preprocessor):
        raise ValueError("Neural bundle and fitted preprocessing metadata differ")
    with neural_execution(device):
        import torch
        from .models.neural import TorchFeatures
        for state in (bundle["model_state"], bundle["mapping_state"]):
            if not isinstance(state, dict) or not state or any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in state.values()):
                raise ValueError("Neural state must contain finite tensors")
        # Shape constructor only. All learned mapping buffers are restored strictly.
        raw = np.stack([bundle["mapping_state"]["raw_mean"].cpu().numpy()] * 2)
        mapping = TorchFeatures(preprocessor, raw)
        mapping.load_state_dict(bundle["mapping_state"], strict=True)
        model = _architecture(bundle["model_id"], preprocessor, settings)
        model.load_state_dict(bundle["model_state"], strict=True)
        model.to(device).eval()
        mapping.to(device).eval()
        return NeuralPredictor(bundle["model_id"], model, mapping, preprocessor, settings, device)


def _fit(X, y, config, settings, model_id, seed, split, epochs, phase, fold, device_report):
    import torch
    from .models.neural import TorchFeatures, cpu_state_dict, shape_penalty
    train, valid = split if split is not None else (np.arange(len(X)), None)
    preprocessor = make_preprocessor(config.injection_mode)
    preprocessor.fit(X.iloc[train])
    raw = imputed_raw(preprocessor, X.iloc[train])
    mapping = TorchFeatures(preprocessor, raw).to(config.device)
    model = _architecture(model_id, preprocessor, settings).to(config.device)
    predictor = NeuralPredictor(model_id, model, mapping, preprocessor, settings, config.device)
    tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=config.device)
    coordinates = ((tensor(raw) - mapping.raw_mean) / mapping.raw_scale).detach()
    targets = tensor(y[train])
    valid_coordinates = None
    if valid is not None:
        valid_raw = imputed_raw(preprocessor, X.iloc[valid])
        valid_coordinates = ((tensor(valid_raw) - mapping.raw_mean) / mapping.raw_scale).detach()
        valid_targets = tensor(y[valid])
    constraints = settings["constraints"]
    constrained = model_id == "NN_PINN_HARD"
    collocation = None
    if constrained and constraints["collocation_count"] and constraints["collocation_weight"]:
        points = collocation_raw(raw, constraints["collocation_count"], task_seed(seed, model_id, "collocation"))
        collocation = ((tensor(points) - mapping.raw_mean) / mapping.raw_scale).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    history, best_state, best_epoch, best_rmse, stale = [], None, None, float("inf"), 0
    def penalty(points):
        leaf = points.detach().clone().requires_grad_(True)
        return shape_penalty(lambda z: model(mapping(z)), leaf, mapping.gas_index,
                             constraints["gas_monotonicity_weight"], constraints["gas_concavity_weight"])
    budget = epochs if epochs is not None else settings["max_epochs"]
    with progress(total=budget, desc=f"{model_id} {phase} fold {fold}", unit="epoch", leave=False) as bar:
        for epoch in range(1, budget + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mse = torch.nn.functional.mse_loss(model(mapping(coordinates)), targets)
            real_penalty = mse.new_zeros(())
            collocation_penalty = mse.new_zeros(())
            if constrained and (constraints["gas_monotonicity_weight"] or constraints["gas_concavity_weight"]):
                if constraints["real_weight"]:
                    real_penalty = penalty(coordinates)
                if collocation is not None:
                    collocation_penalty = penalty(collocation)
            loss = mse + constraints["real_weight"] * real_penalty + constraints["collocation_weight"] * collocation_penalty
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite neural loss at epoch {epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"Nonfinite neural gradient at epoch {epoch}")
            optimizer.step()
            model.eval()
            with torch.no_grad():
                train_mse = torch.nn.functional.mse_loss(model(mapping(coordinates)), targets).item()
                rmse = None if valid is None else torch.nn.functional.mse_loss(model(mapping(valid_coordinates)), valid_targets).sqrt().item()
            if not np.isfinite(train_mse) or (rmse is not None and not np.isfinite(rmse)):
                raise ValueError(f"Nonfinite neural score at epoch {epoch}")
            history.append({"epoch": epoch, "loss_pre_update": loss.item(), "data_mse_pre_update": mse.item(),
                            "real_shape_penalty": real_penalty.item(), "collocation_shape_penalty": collocation_penalty.item(),
                            "train_mse_post_update": train_mse, "inner_validation_rmse": rmse})
            bar.update(1)
            scores = {"train_RMSE_pp": f"{np.sqrt(train_mse):.4f}"}
            if rmse is not None:
                scores["inner_RMSE_pp"] = f"{rmse:.4f}"
            bar.set_postfix(scores, refresh=False)
            if valid is not None:
                if rmse < best_rmse:
                    # Always restore the lowest RMSE; min_delta controls patience only.
                    improvement = best_rmse - rmse
                    best_rmse, best_epoch, best_state = rmse, epoch, cpu_state_dict(model)
                    stale = 0 if improvement > settings["early_stopping"]["min_delta"] else stale + 1
                else:
                    stale += 1
                if stale >= settings["early_stopping"]["patience"]:
                    break
    if valid is not None:
        model.load_state_dict(best_state, strict=True)
    else:
        best_epoch = budget
    details = {"seed": seed, "phase": phase, "fold": fold, "device": device_report,
               "settings": deepcopy(settings), "history": history, "epochs_run": len(history),
               "selected_epochs": best_epoch, "best_inner_rmse": None if valid is None else best_rmse,
               "best_checkpoint_restored": valid is not None,
               "epoch_policy": "fixed_B_selected_budget" if valid is None else "minimum_inner_validation_RMSE",
               "inner_training_positions": train.tolist(), "inner_validation_positions": None if valid is None else valid.tolist(),
               "preprocessing": preprocessing_metadata(preprocessor), "fallback_performed": False,
               "constraints": {"bounded_output": constrained, "PDE_residual": False,
                               "whole_network_monotonicity_guaranteed": False, "permeability_monotonicity": False,
                               "derivative_coordinate": "canonical gas standardized by inner-training raw population std",
                               "collocation": "bootstrap raw covariates; gas uniform inside training range; no expansion",
                               "OOD_clamp": False}}
    return NeuralResult(model_id, "fitted", predictor, details)


def fit_neural(X, y, config, settings, model_id, *, inner_split=None, epochs=None, phase="inner", fold=0):
    """Early-stop inside passed B development, or refit its fixed selected epoch budget.

    No outer-validation argument exists. Fixed-budget fits cannot accept an inner
    split. The caller must supply B-derived epochs for refits; Part 7 will verify
    that provenance. Never use this helper to tune on A.
    """
    settings = validate_neural_settings(settings)
    if model_id not in settings["enabled_models"]:
        raise ValueError("Neural model must be explicitly enabled")
    for column, value in (("source", "OPM_SPE5_simulation"), ("reservoir_family", "B")):
        if column not in X or not X[column].eq(value).all():
            raise ValueError("Neural fitting requires B simulation rows only")
    targets = training_targets(X, y)
    if epochs is not None and (type(epochs) is not int or not 1 <= epochs <= settings["max_epochs"] or inner_split is not None):
        raise ValueError("Fixed epochs must be within max_epochs and cannot use early-stopping rows")
    split_seed = task_seed(config.seed, "split", "neural_inner", fold)
    split = inner_partition(len(X), settings, split_seed, inner_split) if epochs is None else None
    seed = task_seed(config.seed, model_id, phase, fold)
    with neural_execution(config.device) as report:
        import torch
        devices = [] if config.device == "cpu" else [int(config.device.split(":")[1])]
        with threadpool_limits(limits=1), torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)  # only current explicitly selected device
            try:
                result = _fit(X, targets, config, settings, model_id, seed, split, epochs, phase, fold, report)
                result.details["inner_split_seed"] = split_seed if split is not None else None
                return result
            except Exception as exc:
                return NeuralResult(model_id, "failed", details={"seed": seed, "phase": phase, "fold": fold,
                    "device": report, "fallback_performed": False, "error": {"type": type(exc).__name__,
                    "message": str(exc), "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=6))}})


def fit_neural_models(X, y, config, settings, *, selected_epochs=None, inner_split=None, phase="inner", fold=0):
    """Sequential coordinator fits; no joblib pool or DataLoader subprocesses."""
    settings = validate_neural_settings(settings)
    if selected_epochs is not None and (not isinstance(selected_epochs, dict) or set(selected_epochs) != set(settings["enabled_models"]) or inner_split is not None):
        raise ValueError("Refitting requires selected epochs for every enabled model, without an inner split")
    if selected_epochs is not None and any(type(x) is not int or not 1 <= x <= settings["max_epochs"] for x in selected_epochs.values()):
        raise ValueError("Every selected epoch budget must be within max_epochs before refitting")
    return [fit_neural(X, y, config, settings, name, inner_split=inner_split,
                      epochs=None if selected_epochs is None else selected_epochs[name], phase=phase, fold=fold)
            for name in settings["enabled_models"]]

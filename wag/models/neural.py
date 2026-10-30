"""PyTorch architectures and differentiable feature map; bootstrap before import.

Only the bounded output is guaranteed. The positive gas channel and derivative
penalties do NOT guarantee whole-network monotonicity and contain no PDE residual.
Import performs no tensor work, CUDA initialization, fitting or file access.
"""

import numpy as np
import torch
from torch import nn

from ..features import FeatureSpec, preprocessing_metadata


def _layers(d, hidden):
    layers = []
    for width in hidden:
        layers.extend([nn.Linear(d, width), nn.SiLU()])
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, d, hidden=(96, 96, 48)):
        super().__init__()
        self.net = _layers(d, hidden)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HardPINN(nn.Module):
    """Legacy bounded trunk plus nonnegative-slope gas channel, honestly labeled."""

    def __init__(self, d, gas_index=None, hidden=(96, 96, 48), bounds=(-100., 100.)):
        super().__init__()
        self.trunk = _layers(d, hidden)
        self.gas_index = gas_index
        self.lower, self.upper = bounds
        if gas_index is not None:
            self.gas_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.trunk(x).squeeze(-1)
        if self.gas_index is not None:
            z = z + torch.nn.functional.softplus(self.gas_weight) * x[:, self.gas_index]
        return self.lower + (self.upper - self.lower) * torch.sigmoid(z)


class TorchFeatures(nn.Module):
    """Frozen fold-fitted mapping from standardized canonical RAW coordinates.

    Differentiating through this map changes raw gas and its log together. The
    derivative coordinate is gas standardized by its inner-training raw std,
    not an independently varied engineered column or a derivative in field units.
    """

    def __init__(self, preprocessor, raw_training):
        super().__init__()
        metadata = preprocessing_metadata(preprocessor)
        self.mode = metadata["specification"]["injection_mode"]
        self.spec = FeatureSpec(self.mode)
        self.gas_index = len(self.spec.raw_features) - 2
        mean = raw_training.mean(axis=0)
        scale = raw_training.std(axis=0)
        scale[scale == 0] = 1.
        for name, array in (("raw_mean", mean), ("raw_scale", scale),
                            ("mean", metadata["scaler"]["mean"]),
                            ("scale", metadata["scaler"]["scale"])):
            self.register_buffer(name, torch.tensor(np.asarray(array), dtype=torch.float32))
        mask = preprocessor.named_steps["constants"].active_mask_
        self.register_buffer("active", torch.tensor(np.flatnonzero(mask), dtype=torch.long))
        self.register_buffer("mmp", torch.tensor(metadata["fixed_context"]["mmp_full_psi"], dtype=torch.float32))

    def forward(self, coordinates):
        raw = coordinates * self.raw_scale + self.raw_mean
        # Raw order is the versioned FeatureSpec order (9 base + 2 injection).
        extra = [torch.log10(raw[:, 0]), raw[:, 0] * raw[:, 1],
                 raw[:, 2] / raw[:, 3], raw[:, 2] / self.mmp,
                 torch.log1p(raw[:, -2]), torch.log1p(raw[:, -1])]
        engineered = torch.cat([raw, torch.stack(extra, dim=1)], dim=1)
        return (engineered.index_select(1, self.active) - self.mean) / self.scale


def gas_derivatives(predict, coordinates, gas_index):
    """Return first/second gas derivatives, handling constant/linear functions."""
    prediction = predict(coordinates)
    zero = coordinates[:, gas_index] * 0.
    if not prediction.requires_grad:
        return zero, zero
    gradient = torch.autograd.grad(prediction.sum(), coordinates, create_graph=True,
                                   allow_unused=True)[0]
    if gradient is None:
        return zero, zero
    first = gradient[:, gas_index]
    if not first.requires_grad:
        return first, zero
    second = torch.autograd.grad(first.sum(), coordinates, create_graph=True,
                                 allow_unused=True)[0]
    return first, zero if second is None else second[:, gas_index]


def shape_penalty(predict, coordinates, gas_index, monotonicity_weight, concavity_weight):
    """Penalty: negative first derivative / positive second derivative in gas."""
    first, second = gas_derivatives(predict, coordinates, gas_index)
    return (monotonicity_weight * torch.relu(-first).square().mean()
            + concavity_weight * torch.relu(second).square().mean())


def cpu_state_dict(model):
    """Independent CPU snapshots, never mutable references to live training state."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from art.attacks.evasion import AutoProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import PyTorchClassifier


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class NormalizedModelWrapper(nn.Module):
    """
    Wrap a model that expects normalized CIFAR-10 inputs, while exposing
    a [0,1]-input interface to ART.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

        mean = torch.tensor(_CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def normalize_cifar10_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_CIFAR10_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR10_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def build_art_classifier_cifar10(
    model: nn.Module,
    device: torch.device,
    nb_classes: int = 10,
) -> PyTorchClassifier:
    wrapped_model = NormalizedModelWrapper(model).to(device)
    wrapped_model.eval()

    dummy_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=1.0)
    loss = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=loss,
        optimizer=dummy_optimizer,
        input_shape=(3, 32, 32),
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier


def build_fgsm_for_cifar10(
    model: nn.Module,
    device: torch.device,
    norm: str = "linf",
    eps: float = 8.0 / 255.0,
    batch_size: int = 128,
) -> FastGradientMethod:
    classifier = build_art_classifier_cifar10(model, device=device, nb_classes=10)

    norm_value = np.inf if norm.lower() == "linf" else 2

    attacker = FastGradientMethod(
        estimator=classifier,
        norm=norm_value,
        eps=float(eps),
        eps_step=float(eps),
        targeted=False,
        batch_size=int(batch_size),
    )
    return attacker


def build_autopgd_for_cifar10(
    model: nn.Module,
    device: torch.device,
    norm: str = "linf",
    eps: float = 8.0 / 255.0,
    eps_step: Optional[float] = None,
    max_iter: int = 100,
    nb_random_init: int = 5,
    batch_size: int = 128,
    loss_type: str = "cross_entropy",
    verbose: bool = False,
) -> AutoProjectedGradientDescent:
    classifier = build_art_classifier_cifar10(model, device=device, nb_classes=10)

    norm_value = np.inf if norm.lower() == "linf" else 2

    if eps_step is None:
        eps_step = float(eps) / 4.0

    attacker = AutoProjectedGradientDescent(
        estimator=classifier,
        norm=norm_value,
        eps=float(eps),
        eps_step=float(eps_step),
        max_iter=int(max_iter),
        targeted=False,
        nb_random_init=int(nb_random_init),
        batch_size=int(batch_size),
        loss_type=loss_type,
        verbose=verbose,
    )
    return attacker


def generate_adversarial_batch(attacker, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> torch.Tensor:
    x_np = x.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy()

    x_adv_np = attacker.generate(x=x_np, y=y_np)
    x_adv = torch.from_numpy(x_adv_np).to(device=device, dtype=x.dtype)
    return x_adv
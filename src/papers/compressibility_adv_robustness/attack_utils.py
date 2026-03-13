from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from art.attacks.evasion import AutoProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import PyTorchClassifier


_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


def mnist_normalized_clip_values() -> tuple[float, float]:
    lower = (0.0 - _MNIST_MEAN) / _MNIST_STD
    upper = (1.0 - _MNIST_MEAN) / _MNIST_STD
    return float(lower), float(upper)


def scale_l2_budget_to_normalized_space(eps: float) -> float:
    return float(eps) / _MNIST_STD


def build_art_classifier(
    model: nn.Module,
    device: torch.device,
    nb_classes: int = 2,
) -> PyTorchClassifier:
    model.eval()

    # ART requires an optimizer object for the PyTorchClassifier constructor.
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loss = nn.CrossEntropyLoss()
    clip_values = mnist_normalized_clip_values()

    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=dummy_optimizer,
        input_shape=(1, 28, 28),
        nb_classes=nb_classes,
        clip_values=clip_values,
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier


def build_autopgd_l2_for_mnist(
    model: nn.Module,
    device: torch.device,
    eps_l2_original_space: float = 2, #before it was 0.125
    eps_step_l2_original_space: float | None = None,
    max_iter: int = 100,
    nb_random_init: int = 5,
    batch_size: int = 128,
    loss_type: str = "cross_entropy",
    verbose: bool = False,
) -> AutoProjectedGradientDescent:
    if eps_step_l2_original_space is None:
        eps_step_l2_original_space = eps_l2_original_space / 4.0

    classifier = build_art_classifier(model, device=device, nb_classes=2)

    attacker = AutoProjectedGradientDescent(
        estimator=classifier,
        norm=2,
        eps=scale_l2_budget_to_normalized_space(eps_l2_original_space),
        eps_step=scale_l2_budget_to_normalized_space(eps_step_l2_original_space),
        max_iter=int(max_iter),
        targeted=False,
        nb_random_init=int(nb_random_init),
        batch_size=int(batch_size),
        loss_type=loss_type,
        verbose=verbose,
    )
    return attacker


def build_fgsm_l2_for_mnist(
    model: nn.Module,
    device: torch.device,
    eps_l2_original_space: float = 2, #before it was 0.125
    batch_size: int = 128,
) -> FastGradientMethod:
    classifier = build_art_classifier(model, device=device, nb_classes=2)

    attacker = FastGradientMethod(
        estimator=classifier,
        norm=2,
        eps=scale_l2_budget_to_normalized_space(eps_l2_original_space),
        targeted=False,
        batch_size=int(batch_size),
    )
    return attacker



def generate_adversarial_batch(
    attacker,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x_np = x.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy()

    x_adv_np = attacker.generate(x=x_np, y=y_np)
    x_adv = torch.from_numpy(x_adv_np).to(device=device, dtype=x.dtype)
    return x_adv
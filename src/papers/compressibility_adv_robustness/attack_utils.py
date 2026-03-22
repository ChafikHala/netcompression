from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from art.attacks.evasion import AutoProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import PyTorchClassifier


_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def mnist_normalized_clip_values() -> tuple[float, float]:
    lower = (0.0 - _MNIST_MEAN) / _MNIST_STD
    upper = (1.0 - _MNIST_MEAN) / _MNIST_STD
    return float(lower), float(upper)


def scale_l2_budget_to_normalized_space(eps: float, dataset: str) -> float:
    if dataset == "mnist":
        return float(eps) / _MNIST_STD
    if dataset == "cifar10":
        return float(eps) / float(np.mean(_CIFAR10_STD))


def cifar10_normalized_clip_values() -> tuple[np.ndarray, np.ndarray]:
    lower = (np.array([0.0, 0.0, 0.0], dtype=np.float32) - np.array(_CIFAR10_MEAN, dtype=np.float32)) / np.array(_CIFAR10_STD, dtype=np.float32)
    upper = (np.array([1.0, 1.0, 1.0], dtype=np.float32) - np.array(_CIFAR10_MEAN, dtype=np.float32)) / np.array(_CIFAR10_STD, dtype=np.float32)
    return lower, upper


def scale_linf_budget_to_normalized_space(eps: float) -> np.ndarray:
    arr = np.array([eps / s for s in _CIFAR10_STD], dtype=np.float32)
    return arr.reshape((1, 3, 1, 1))


def build_art_classifier(
    model: nn.Module,
    device: torch.device,
    nb_classes: int = 2,
    dataset : str = "mnist"
) -> PyTorchClassifier:
    model.eval()

    # ART requires an optimizer object for the PyTorchClassifier constructor.
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loss = nn.CrossEntropyLoss()
    clip_values = None
    if dataset == "mnist":
        clip_values = mnist_normalized_clip_values()
    if dataset == "cifar10":
        clip_values = cifar10_normalized_clip_values()

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
        eps=scale_l2_budget_to_normalized_space(eps_l2_original_space, "mnist"),
        eps_step=scale_l2_budget_to_normalized_space(eps_step_l2_original_space, "mnist"),
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
        eps=scale_l2_budget_to_normalized_space(eps_l2_original_space, "mnist"),
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


def build_autopgd_l2_for_cifar10(
    model: nn.Module,
    device: torch.device,
    eps_l2_original_space: float = 0.5,
    eps_step_l2_original_space: float | None = None,
    max_iter: int = 100,
    nb_random_init: int = 5,
    batch_size: int = 128,
    loss_type: str = "cross_entropy",
    verbose: bool = False,
) -> AutoProjectedGradientDescent:
    if eps_step_l2_original_space is None:
        eps_step_l2_original_space = eps_l2_original_space / 4.0

    classifier = build_art_classifier(model, device=device, nb_classes=10)

    attacker = AutoProjectedGradientDescent(
        estimator=classifier,
        norm=2,
        eps=scale_l2_budget_to_normalized_space(eps_l2_original_space, "cifar10"),
        eps_step=scale_l2_budget_to_normalized_space(eps_step_l2_original_space, "cifar10"),
        max_iter=int(max_iter),
        targeted=False,
        nb_random_init=int(nb_random_init),
        batch_size=int(batch_size),
        loss_type=loss_type,
        verbose=verbose,
    )
    return attacker


# --------------------------------------------------
# APGD-Linf for CIFAR-10
# --------------------------------------------------
def build_autopgd_linf_for_cifar10(
    model: nn.Module,
    device: torch.device,
    eps_linf_original_space: float = 8 / 255,
    eps_step_linf_original_space: float | None = None,
    max_iter: int = 100,
    nb_random_init: int = 5,
    batch_size: int = 128,
    loss_type: str = "cross_entropy",
    verbose: bool = False,
) -> AutoProjectedGradientDescent:
    if eps_step_linf_original_space is None:
        eps_step_linf_original_space = eps_linf_original_space / 4.0

    classifier = build_art_classifier(model, device=device, nb_classes=10)

    attacker = AutoProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=scale_linf_budget_to_normalized_space(eps_linf_original_space),
        eps_step=scale_linf_budget_to_normalized_space(eps_step_linf_original_space),
        max_iter=int(max_iter),
        targeted=False,
        nb_random_init=int(nb_random_init),
        batch_size=int(batch_size),
        loss_type=loss_type,
        verbose=verbose,
    )
    return attacker


# --------------------------------------------------
# FGSM-L2 for CIFAR-10
# --------------------------------------------------
def build_fgsm_l2_for_cifar10(
    model: nn.Module,
    device: torch.device,
    eps_l2_original_space: float = 0.5,
    batch_size: int = 128,
) -> FastGradientMethod:
    classifier = build_art_classifier(model, device=device, nb_classes=10)

    attacker = FastGradientMethod(
        estimator=classifier,
        norm=2,
        eps=scale_l2_budget_to_normalized_space(eps_l2_original_space, "cifar10"),
        targeted=False,
        batch_size=int(batch_size),
    )
    return attacker


# --------------------------------------------------
# FGSM-Linf for CIFAR-10
# --------------------------------------------------
def build_fgsm_linf_for_cifar10(
    model: nn.Module,
    device: torch.device,
    eps_linf_original_space: float = 2 / 255,
    eps_step_linf_original_space: float | None = None,
    batch_size: int = 128,
) -> FastGradientMethod:
    if eps_step_linf_original_space is None:
        eps_step_linf_original_space = eps_linf_original_space

    classifier = build_art_classifier(model, device=device, nb_classes=10)

    attacker = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=scale_linf_budget_to_normalized_space(eps_linf_original_space),
        eps_step=scale_linf_budget_to_normalized_space(eps_step_linf_original_space),
        targeted=False,
        batch_size=int(batch_size),
    )
    return attacker



from __future__ import annotations

import torch
import torch.nn as nn


def get_single_hidden_layer_weight(model: nn.Module) -> torch.nn.Parameter:
    """
    Return the weight matrix W of the unique hidden layer for the paper's
    1-hidden-layer fully connected network.

    Expected architecture:
        Flatten -> Linear(in, hidden) -> ReLU -> Linear(hidden, num_classes)

    We regularize and renormalize the first Linear layer only.
    """
    if not hasattr(model, "net"):
        raise ValueError("Expected model to have attribute `net` (as in src.models.fcn.FCN).")

    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]

    if len(linear_layers) != 2:
        raise ValueError(
            "Expected exactly two Linear layers for a 1-hidden-layer FCN "
            f"(found {len(linear_layers)})."
        )

    hidden_linear = linear_layers[0]
    return hidden_linear.weight


def nuclear_norm_penalty(weight: torch.Tensor) -> torch.Tensor:
    """
    Nuclear norm = sum of singular values.
    This is the spectral compressibility regularizer used in the experiment.
    """
    return torch.linalg.matrix_norm(weight, ord="nuc")


@torch.no_grad()
def frobenius_normalize_(weight: torch.Tensor, target_fro_norm: float) -> None:
    """
    In-place rescaling of the matrix so that ||W||_F = target_fro_norm.

    If the current norm is zero (or numerically tiny), the function does nothing
    to avoid division instability.
    """
    current = torch.linalg.matrix_norm(weight, ord="fro")
    if current <= 1e-12:
        return
    weight.mul_(float(target_fro_norm) / current)


@torch.no_grad()
def capture_frobenius_norm(weight: torch.Tensor) -> float:
    """
    Capture the reference Frobenius norm of W, typically once at initialization,
    so that later iterations can be renormalized back to this fixed value.
    """
    return float(torch.linalg.matrix_norm(weight, ord="fro").item())


@torch.no_grad()
def collect_matrix_stats(weight: torch.Tensor) -> dict[str, float]:
    """
    Optional diagnostics for logging/debugging.
    """
    fro = float(torch.linalg.matrix_norm(weight, ord="fro").item())
    nuc = float(torch.linalg.matrix_norm(weight, ord="nuc").item())
    return {
        "fro_norm": fro,
        "nuclear_norm": nuc,
    }
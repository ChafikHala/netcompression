from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR

from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.device import get_device

from src.data.datasets import build_datasets
from src.data.dataloaders import build_dataloaders

from src.models.factory import build_model
from src.core.trainer import train


def _build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt = cfg.optimizer
    opt_type = opt.type.lower()

    params = model.parameters()
    lr = float(opt.lr)

    if opt_type == "sgd":
        momentum = float(getattr(opt, "momentum", 0.0))
        weight_decay = float(getattr(opt, "weight_decay", 0.0))
        nesterov = bool(getattr(opt, "nesterov", False))

        return SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if opt_type == "adam":
        betas = tuple(getattr(opt, "betas", (0.9, 0.999)))
        eps = float(getattr(opt, "eps", 1e-8))
        weight_decay = float(getattr(opt, "weight_decay", 0.0))

        return Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    if opt_type == "adamw":
        betas = tuple(getattr(opt, "betas", (0.9, 0.999)))
        eps = float(getattr(opt, "eps", 1e-8))
        weight_decay = float(getattr(opt, "weight_decay", 1e-2))  # important default

        return AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")


def _build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    sch = cfg.scheduler
    sch_type = sch.type.lower()

    if sch_type == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=[int(m) for m in sch.milestones],
            gamma=float(sch.gamma),
        )

    if sch_type in {"none", "null"}:
        return None

    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed_everything(int(cfg.experiment.seed))
    device = get_device(getattr(cfg.experiment, "device", "auto"))
    print("Using device:", device)

    # Data
    bundle = build_datasets(cfg)
    train_loader, val_loader, _ = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)


    # Model
    model = build_model(cfg, num_classes=bundle.num_classes)

    # Optim / sched
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    # Train
    result = train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    print(
        f"Done. Best {cfg.checkpoint.monitor}={result.best_metric:.4f} "
        f"at epoch {result.best_epoch}."
    )


if __name__ == "__main__":
    main()

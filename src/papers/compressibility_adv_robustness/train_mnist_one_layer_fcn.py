from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.core.checkpoint import CheckpointPayload, save_checkpoint
from src.core.evaluator import evaluate
from src.data.dataloaders import build_dataloaders
from src.data.datasets import build_datasets
from src.models.fcn import FCN
from src.papers.compressibility_adv_robustness.regularization import (
    collect_matrix_stats,
    frobenius_normalize_,
    get_single_hidden_layer_weight,
    nuclear_norm_penalty,
)
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import seed_everything


@dataclass
class EarlyStoppingState:
    best_val_loss: float = math.inf
    best_epoch: int = -1
    epochs_without_improvement: int = 0


def _alpha_to_name(alpha: float) -> str:
    if float(alpha).is_integer():
        return str(int(alpha))
    return str(alpha).replace(".", "p")


def _set_nested_attr(obj: Any, path: list[str], value: Any) -> None:
    cur = obj
    for key in path[:-1]:
        cur = getattr(cur, key)
    setattr(cur, path[-1], value)


def _build_model(num_classes: int = 2) -> nn.Module:
    return FCN(
        input_shape=[1, 28, 28],
        hidden_dims=[400],
        num_classes=num_classes,
        dropout=0.0,
        bias=False,
    )

def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_jsonl_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _prepare_cfg(cfg, alpha: float):
    """
    Override only what is needed for this paper-specific experiment.
    """
    cfg = load_config_dict_like(cfg)

    _set_nested_attr(cfg, ["experiment", "name"], f"model_one_layer_FC_alpha_{_alpha_to_name(alpha)}")
    _set_nested_attr(cfg, ["dataset", "name"], "mnist")
    _set_nested_attr(cfg, ["dataset", "num_classes"], 2)
    _set_nested_attr(cfg, ["dataset", "binary"], True)
    _set_nested_attr(cfg, ["dataset", "val_fraction"], 0.05)

    _set_nested_attr(cfg, ["model", "name"], "fcn")
    _set_nested_attr(cfg, ["model", "input_shape"], [1, 28, 28])
    _set_nested_attr(cfg, ["model", "hidden_dims"], [400])
    _set_nested_attr(cfg, ["model", "dropout"], 0.0)

    _set_nested_attr(cfg, ["optimizer", "type"], "adamw")
    _set_nested_attr(cfg, ["optimizer", "lr"], 1e-3)
    _set_nested_attr(cfg, ["optimizer", "weight_decay"], 1e-2)

    _set_nested_attr(cfg, ["training", "criterion"], "cross_entropy")
    _set_nested_attr(cfg, ["training", "label_smoothing"], 0.0)

    _set_nested_attr(cfg, ["model", "bias"], False)

    return cfg


def load_config_dict_like(cfg):
    """
    Defensive helper:
    - if cfg is already your Config object, keep it
    - otherwise turn a plain dict into the same Config type by round-tripping
      through the existing constructor indirectly
    """
    if hasattr(cfg, "to_dict"):
        return cfg

    class _SimpleConfig:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    v = _SimpleConfig(v)
                setattr(self, k, v)

        def to_dict(self):
            out = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "to_dict"):
                    v = v.to_dict()
                out[k] = v
            return out

    return _SimpleConfig(cfg)


def _save_best_checkpoint(
    *,
    path: Path,
    epoch: int,
    best_val_loss: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg,
) -> None:
    payload = CheckpointPayload(
        epoch=epoch,
        best_metric=float(best_val_loss),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        config=cfg.to_dict(),
    )
    save_checkpoint(path, payload)


def train_one_alpha(cfg, alpha: float, seed: Optional[int] = None) -> dict:
    cfg = load_config_dict_like(cfg)
    if seed is not None:
        _set_nested_attr(cfg, ["experiment", "seed"], int(seed))

    seed_everything(int(cfg.experiment.seed))
    device = get_device(getattr(cfg.experiment, "device", "auto"))
    print("Using device:", device)

    cfg = _prepare_cfg(cfg, alpha)

    # Data
    bundle = build_datasets(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, bundle.train, bundle.val, bundle.test, device
    )

    # Model / optimizer / loss
    model = _build_model(num_classes=bundle.num_classes).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
    )
    criterion = nn.CrossEntropyLoss()


    # Apply Frobenius normalization only when alpha > 0
    # use_frobenius_normalization = float(alpha) > 0.0
    use_frobenius_normalization = True

    target_fro_norm = 11.5

    if use_frobenius_normalization:
        weight = get_single_hidden_layer_weight(model)
        frobenius_normalize_(weight, target_fro_norm)


    # Output structure
    output_root = Path(getattr(cfg.experiment, "output_dir", "outputs"))
    exp_name = cfg.experiment.name
    run_name = f"{exp_name}_seed_{cfg.experiment.seed}"
    run_dir = output_root / "compressibility_adv_robustness" / run_name
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    _save_json(run_dir / "config.json", cfg.to_dict())

    best_ckpt_path = ckpt_dir / "best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"
    metrics_jsonl_path = logs_dir / "metrics.jsonl"

    max_epochs = int(cfg.training.epochs)
    patience = int(cfg.training.early_stopping_patience)

    early = EarlyStoppingState()

    print(f"[Experiment] {exp_name}")
    print(f"[Alpha] {alpha}")
    print(f"[Run dir] {run_dir}")
    print(f"[Target Frobenius norm] {target_fro_norm:.6f}")
    print(f"[Early stopping patience] {patience}")

    history: list[dict] = []

    for epoch in range(max_epochs):
        model.train()

        running_total = 0
        running_correct = 0

        running_ce_loss = 0.0
        running_nuclear_penalty = 0.0
        running_total_loss = 0.0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            ce_loss = criterion(logits, y)

            weight = get_single_hidden_layer_weight(model)
            if alpha:
                nuc_pen = nuclear_norm_penalty(weight)
            else:
                nuc_pen = torch.tensor(0.)

            loss = ce_loss + float(alpha) * nuc_pen

            loss.backward()
            optimizer.step()

            # Enforce fixed Frobenius norm after each optimizer step
            weight = get_single_hidden_layer_weight(model)
            
            # if alpha
            frobenius_normalize_(weight, target_fro_norm)

            bs = x.size(0)
            preds = logits.argmax(dim=1)

            running_total += bs
            running_correct += int((preds == y).sum().item())
            running_ce_loss += float(ce_loss.item()) * bs
            running_nuclear_penalty += float(nuc_pen.item()) * bs
            running_total_loss += float(loss.item()) * bs

        train_ce = running_ce_loss / max(running_total, 1)
        train_nuc = running_nuclear_penalty / max(running_total, 1)
        train_total_loss = running_total_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_res = evaluate(model, val_loader, criterion, device)
        # test_res = evaluate(model, test_loader, criterion, device)

        matrix_stats = collect_matrix_stats(get_single_hidden_layer_weight(model))

        record = {
            "epoch": epoch + 1,
            "alpha": float(alpha),
            "train_ce_loss": train_ce,
            "train_nuclear_norm": train_nuc,
            "train_total_loss": train_total_loss,
            "train_accuracy": train_acc,
            "val_loss": float(val_res.loss),
            "val_accuracy": float(val_res.accuracy),
            # "test_loss": float(test_res.loss),
            # "test_accuracy": float(test_res.accuracy),
            "fro_norm": matrix_stats["fro_norm"],
            "nuclear_norm": matrix_stats["nuclear_norm"],
        }
        history.append(record)
        _save_jsonl_line(metrics_jsonl_path, record)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"train_ce={train_ce:.4f} | "
            f"train_nuc={train_nuc:.4f} | "
            f"train_total={train_total_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_res.loss:.4f} | "
            f"val_acc={val_res.accuracy:.4f} | "
            # f"test_acc={test_res.accuracy:.4f} | "
            f"fro={matrix_stats['fro_norm']:.4f} | "
            f"nuc={matrix_stats['nuclear_norm']:.4f}"
        )

        # Best model selection and early stopping are based on validation loss
        if val_res.loss < early.best_val_loss:
            early.best_val_loss = float(val_res.loss)
            early.best_epoch = epoch
            early.epochs_without_improvement = 0

            _save_best_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                best_val_loss=early.best_val_loss,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )
        else:
            early.epochs_without_improvement += 1

        # Always save last checkpoint
        last_payload = CheckpointPayload(
            epoch=epoch,
            best_metric=float(early.best_val_loss),
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None,
            config=cfg.to_dict(),
        )
        save_checkpoint(last_ckpt_path, last_payload)

        if early.epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best val_loss={early.best_val_loss:.6f} at epoch {early.best_epoch+1}."
            )
            break

    summary = {
        "experiment_name": exp_name,
        "alpha": float(alpha),
        "seed": int(cfg.experiment.seed),
        "best_val_loss": float(early.best_val_loss),
        "best_epoch": int(early.best_epoch + 1),
        "last_epoch": int(len(history)),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "target_fro_norm": float(target_fro_norm),
        "history": history,
    }

    _save_json(run_dir / "summary.json", summary)

    print(
        f"Done. alpha={alpha} | "
        f"best_val_loss={early.best_val_loss:.6f} | "
        f"best_epoch={early.best_epoch+1}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, required=False, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_one_alpha(cfg, alpha=float(args.alpha), seed=args.seed)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.core.checkpoint import save_checkpoint, CheckpointPayload, save_best_if_needed
from src.core.evaluator import evaluate
from src.utils.paths import make_run_paths


@dataclass(frozen=True)
class TrainResult:
    best_metric: float
    best_epoch: int
    last_epoch: int
    run_dir: str


def train(
    *,
    cfg,
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> TrainResult:
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.training.label_smoothing))

    use_amp = bool(cfg.training.mixed_precision) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    paths = make_run_paths(cfg)

    best_path = paths.checkpoints_dir / "best.pt"
    last_path = paths.checkpoints_dir / "last.pt"
    metrics_path = paths.logs_dir / "metrics.jsonl"

    print_every = int(getattr(cfg.logging, "print_every_steps", 100))
    save_epoch_metrics = bool(getattr(cfg.logging, "save_epoch_metrics", True))
    save_batch_logs = bool(getattr(cfg.logging, "save_batch_logs", False))

    monitor = cfg.checkpoint.monitor
    mode = cfg.checkpoint.mode
    save_last = bool(getattr(cfg.checkpoint, "save_last", True))
    save_every = int(getattr(cfg.checkpoint, "save_every_epochs", 0))

    best_metric = float("-inf") if mode == "max" else float("inf")
    best_epoch = -1
    global_step = 0

    print(f"[Run] {paths.run_dir}")
    print(f"[Device] {device} | AMP={use_amp}")
    print(f"[Checkpoint] monitor={monitor} mode={mode} save_every_epochs={save_every}")

    for epoch in range(int(cfg.training.epochs)):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            preds = logits.argmax(dim=1)
            running_correct += int((preds == y).sum().item())
            running_total += int(bs)

            global_step += 1

            if print_every > 0 and (step % print_every == 0):
                train_loss = running_loss / max(running_total, 1)
                train_acc = running_correct / max(running_total, 1)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch+1}/{cfg.training.epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"loss={train_loss:.4f} acc={train_acc:.4f} lr={lr:.5g}"
                )

            # optional: if you really want batch logs, keep them tiny
            if save_batch_logs:
                batch_record = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss_running": running_loss / max(running_total, 1),
                    "train_accuracy_running": running_correct / max(running_total, 1),
                }
                with open(paths.logs_dir / "batch.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(batch_record) + "\n")

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        # Validation
        val_res = evaluate(model, val_loader, criterion, device)
        metrics = {"val_loss": val_res.loss, "val_accuracy": val_res.accuracy}
        metric_value = float(metrics[monitor])

        # Save best
        new_best = save_best_if_needed(
            path=best_path,
            epoch=epoch,
            metric_value=metric_value,
            best_metric=best_metric,
            mode=mode,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config_dict=cfg.to_dict(),
        )
        if new_best != best_metric:
            best_metric = new_best
            best_epoch = epoch

        # Save periodic checkpoint (optional)
        if save_every > 0 and ((epoch + 1) % save_every == 0):
            periodic_path = paths.checkpoints_dir / f"epoch_{epoch+1:04d}.pt"
            payload = CheckpointPayload(
                epoch=epoch,
                best_metric=float(best_metric),
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                config=cfg.to_dict(),
            )
            save_checkpoint(periodic_path, payload)

        # Save last checkpoint
        if save_last:
            payload = CheckpointPayload(
                epoch=epoch,
                best_metric=float(best_metric),
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                config=cfg.to_dict(),
            )
            save_checkpoint(last_path, payload)

        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch+1:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_res.loss:.4f} val_acc={val_res.accuracy:.4f} | "
            f"best={best_metric:.4f} (epoch {best_epoch+1 if best_epoch>=0 else -1}) | "
            f"lr={lr:.5g} | {epoch_time:.1f}s"
        )

        if save_epoch_metrics:
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_res.loss,
                "val_accuracy": val_res.accuracy,
                "best_metric": best_metric,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
                "lr": lr,
                "epoch_time_sec": epoch_time,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    return TrainResult(
        best_metric=best_metric,
        best_epoch=best_epoch,
        last_epoch=int(cfg.training.epochs),
        run_dir=str(paths.run_dir),
    )

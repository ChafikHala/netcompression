from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader


def build_dataloaders(cfg, train_ds, val_ds, test_ds, device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dl_cfg = cfg.dataloader
    pin = bool(cfg.dataloader.pin_memory) and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(dl_cfg.batch_size),
        shuffle=True,
        num_workers=int(dl_cfg.num_workers),
        pin_memory=pin,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(dl_cfg.batch_size),
        shuffle=False,
        num_workers=int(dl_cfg.num_workers),
        pin_memory=pin,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(dl_cfg.batch_size),
        shuffle=False,
        num_workers=int(dl_cfg.num_workers),
        pin_memory=pin,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

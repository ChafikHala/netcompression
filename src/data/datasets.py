from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)

_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


@dataclass(frozen=True)
class DatasetBundle:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    num_classes: int


def _build_cifar10_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    aug = cfg.dataset.augmentation

    train_tfms = []
    test_tfms = []

    if getattr(aug, "random_crop", 0) and aug.random_crop > 0:
        train_tfms.append(transforms.RandomCrop(32, padding=int(aug.random_crop)))
    if getattr(aug, "horizontal_flip", False):
        train_tfms.append(transforms.RandomHorizontalFlip())

    train_tfms.append(transforms.ToTensor())
    test_tfms.append(transforms.ToTensor())

    if getattr(aug, "normalize", True):
        norm = transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)
        train_tfms.append(norm)
        test_tfms.append(norm)

    return transforms.Compose(train_tfms), transforms.Compose(test_tfms)


def _build_mnist_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    aug = cfg.dataset.augmentation

    train_tfms = [transforms.ToTensor()]
    test_tfms = [transforms.ToTensor()]

    if getattr(aug, "normalize", True):
        norm = transforms.Normalize(_MNIST_MEAN, _MNIST_STD)
        train_tfms.append(norm)
        test_tfms.append(norm)

    return transforms.Compose(train_tfms), transforms.Compose(test_tfms)


def _split_train_val(train_ds, val_fraction: float, seed: int):
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n_total = len(train_ds)
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(int(seed))
    train_split, val_split = random_split(train_ds, [n_train, n_val], generator=gen)
    return train_split, val_split


def build_datasets(cfg) -> DatasetBundle:
    name = cfg.dataset.name.lower()
    root = cfg.dataset.root

    # default split params (works even if you don't add them to YAML yet)
    val_fraction = float(getattr(cfg.dataset, "val_fraction", 0.1))
    seed = int(cfg.experiment.seed)

    if name == "cifar10":
        train_tfm, test_tfm = _build_cifar10_transforms(cfg)

        full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tfm)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tfm)

        train_ds, val_ds = _split_train_val(full_train, val_fraction=val_fraction, seed=seed)
        return DatasetBundle(train=train_ds, val=val_ds, test=test_ds, num_classes=10)

    if name == "mnist":
        train_tfm, test_tfm = _build_mnist_transforms(cfg)

        full_train = datasets.MNIST(root=root, train=True, download=True, transform=train_tfm)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=test_tfm)

        train_ds, val_ds = _split_train_val(full_train, val_fraction=val_fraction, seed=seed)
        return DatasetBundle(train=train_ds, val=val_ds, test=test_ds, num_classes=10)

    raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

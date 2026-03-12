from __future__ import annotations

import random
from typing import Dict

from torch.utils.data import Dataset, Subset



class NoisySubset(Dataset):
    """Subset that overrides labels for a fixed set of indices."""

    def __init__(self, subset: Subset, corrupted_labels: Dict[int, int]):
        self.dataset = subset.dataset
        self.indices = list(subset.indices)
        self.corrupted_labels = dict(corrupted_labels)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        actual_index = self.indices[index]
        sample, original_label = self.dataset[actual_index]
        label = self.corrupted_labels.get(actual_index, original_label)
        return sample, label


def _get_original_label(dataset: Dataset, index: int) -> int:
    for attr in ("targets", "labels"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return int(value[index])
    _, label = dataset[index]
    return int(label)


def apply_label_noise(
    subset: Subset,
    *,
    num_classes: int,
    noise_fraction: float,
    seed: int,
) -> Dataset:
    if not isinstance(subset, Subset):
        raise TypeError("Label noise can only be applied to torch.utils.data.Subset")
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1, got {num_classes}")
    if not (0.0 <= noise_fraction <= 1.0):
        raise ValueError(f"noise_fraction must be in [0, 1], got {noise_fraction}")
    if noise_fraction == 0.0 or len(subset) == 0:
        return subset

    total = len(subset)
    num_noisy = min(int(round(noise_fraction * total)), total)
    if num_noisy == 0:
        return subset

    rng = random.Random(int(seed))
    positions = rng.sample(range(total), num_noisy)
    indices = list(subset.indices)

    corrupted_labels: Dict[int, int] = {}
    for pos in positions:
        dataset_index = int(indices[pos])
        original_label = _get_original_label(subset.dataset, dataset_index)
        candidates = [cls for cls in range(num_classes) if cls != original_label]
        corrupted_labels[dataset_index] = rng.choice(candidates)

    return NoisySubset(subset, corrupted_labels)

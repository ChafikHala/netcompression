import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Global seed.
        deterministic: If True, forces deterministic algorithms (slower).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch versions might not support this fully.
            pass
    else:
        torch.backends.cudnn.benchmark = True

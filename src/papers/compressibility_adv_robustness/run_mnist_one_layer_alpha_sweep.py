from __future__ import annotations

import argparse
import copy
import numpy as np

from src.papers.compressibility_adv_robustness.train_mnist_one_layer_fcn import train_one_alpha
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-alphas", type=int, default=15)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=3e-1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    alphas = np.geomspace(args.alpha_min, args.alpha_max, args.n_alphas)

    print("Alphas:", [float(a) for a in alphas])
    print("Seeds:", args.seeds)

    for alpha in alphas:
        for seed in args.seeds:
            print("\n" + "=" * 80)
            print(f"Training alpha={float(alpha):.8f}, seed={seed}")
            print("=" * 80)
            cfg = copy.deepcopy(base_cfg)
            train_one_alpha(cfg, alpha=float(alpha), seed=int(seed))


if __name__ == "__main__":
    main()
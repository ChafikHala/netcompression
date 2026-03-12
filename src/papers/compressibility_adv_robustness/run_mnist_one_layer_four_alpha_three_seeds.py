from __future__ import annotations

import argparse
import copy

from src.papers.compressibility_adv_robustness.train_mnist_one_layer_fcn import train_one_alpha
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.05])
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    for alpha in args.alphas:
        for seed in args.seeds:
            print("\n" + "=" * 80)
            print(f"Training alpha={alpha}, seed={seed}")
            print("=" * 80)
            cfg = copy.deepcopy(base_cfg)
            train_one_alpha(cfg, alpha=float(alpha), seed=int(seed))


if __name__ == "__main__":
    main()
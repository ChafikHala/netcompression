#!/bin/bash
#SBATCH --job-name=comp_robust_mnist
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=albatros
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e

cd /users/eleves-a/2022/rida.assalouh/ADL/netcompression || exit 1
mkdir -p slurm_logs

echo "Start time: $(date)"
echo "Host: $(hostname)"

nvidia-smi

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -m src.papers.compressibility_adv_robustness.run_cifar10_wide_resnet_alpha_sweep \
  --config configs/papers/compressibility_adv_robustness/cifar10_wrn16_2.yaml \
  --compressibility neuron \
  --n-betas 10 \
  --beta-min 1e-5 \
  --beta-max 3e-1 \
  --beta-grid geom \
  --seeds 0 1 2

echo "End time: $(date)"
echo "Job finished"

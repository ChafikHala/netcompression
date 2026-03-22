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


python src/experiments/overfitting_harder_to_compress.py \
  --data-root ./data \
  --output-dir ./outputs/noise_pruning_cifar10_wrn16_weight_decay \
  --epochs 100 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --nesterov \
  --seed 0 \
  --alphas 0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
  --pruning-ratios 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99

  

echo "End time: $(date)"
echo "Job finished"

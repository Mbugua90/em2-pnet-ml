#!/bin/bash
#SBATCH --job-name=cnn_lopo_full
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32000
#SBATCH --output=cnn_lopo_full_%j.log
#SBATCH --error=cnn_lopo_full_%j.err

# STAGE 2: Full run -- all 70 folds (6 Pneumonia + 64 COPD patients).
# Only run this AFTER the debug run (train_cnn_debug.sbatch) has completed
# successfully and confirmed the pipeline works end-to-end.
# Uses gpulong (up to 4 days) rather than gpu1 (24h cap) as a safety margin
# for 70 sequential model trainings.
# Submit with: sbatch train_cnn_full.sbatch
# Monitor with: squeue -u $USER
# This job will keep running even if your terminal disconnects.

module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM

cd /scratch/lustre/users/francisco.ngigi24/icbhi_data

echo "Job started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 train_cnn_lopo.py icbhi_cnn_features.npz

echo "Job finished at $(date)"

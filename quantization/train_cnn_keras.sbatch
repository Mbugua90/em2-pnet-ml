#!/bin/bash
#SBATCH --job-name=cnn_keras_ptq
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --output=cnn_keras_ptq_%j.log
#SBATCH --error=cnn_keras_ptq_%j.err

# EM2-PNET Objective 3: CNN branch Keras reimplementation, LOPO-CV
# validation, final model training, and PTQ.
#
# CPU-only (normal partition): GPU acceleration was not available due to
# a network issue downloading the CUDA/cuDNN libraries on this cluster
# (consistent truncation at ~720MB/770MB across 51 retry attempts,
# suggesting a network-level block on large file transfers rather than a
# transient connectivity issue). The 6-fold debug run completed in ~2m20s
# on CPU, confirming the full 70-fold run will comfortably finish well
# within this job's 2-hour limit without GPU acceleration.
#
# Debug run already completed successfully and confirmed the pipeline
# works end-to-end -- this submission runs the FULL 70-fold validation
# plus Stage 2 (final model + PTQ).

module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/users/francisco.ngigi24/tf_env/bin/activate

cd /scratch/lustre/users/francisco.ngigi24/icbhi_data

echo "Job started on $(hostname) at $(date)"

python3 train_cnn_keras_ptq.py icbhi_cnn_features.npz

echo "Job finished at $(date)"

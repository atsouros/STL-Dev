#!/bin/bash
#SBATCH --job-name=micronifty
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0298
#SBATCH --output=micronifty.log
#SBATCH --error=micronifty.err

set -euo pipefail

# Use all CPUs allocated by Slurm for CPU-side work done by NumPy/NIFTy/Torch.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Make CUDA failures synchronous/easier to debug. Comment out for maximal speed.
export CUDA_LAUNCH_BLOCKING=1

# Activate your environment here if needed, for example:
# source ~/.bashrc
# conda activate nifty8

python -u microsample.py --device cuda:0 --dtype float64

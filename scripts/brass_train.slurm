#!/bin/bash
#SBATCH --job-name=gpn_4gpu
#SBATCH --account=imanol-pib
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/faststorage/project/imanol-pib/people/imanolhpc/brass_%j.log
#SBATCH --error=/faststorage/project/imanol-pib/people/imanolhpc/brass_%j.err

# Conda Launch
source ~/.bashrc
conda activate gpn-gpu


export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# PCIe Hang Fix gn-1002
export NCCL_P2P_DISABLE=1


export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

# CPU Nr
export OMP_NUM_THREADS=4

# Working Directory
cd /faststorage/project/imanol-pib/people/imanolhpc/gpn

# Random Port
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# Launch with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    -m gpn.cli ss train recipes/gpn_training/4gpu.yaml

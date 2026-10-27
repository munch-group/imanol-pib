#!/bin/bash
#SBATCH --job-name=gpn_human_sm
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=imanol-pib
#SBATCH --output=/faststorage/project/imanol-pib/people/imanolhpc/gpn/human_sm_train_%j.log
#SBATCH --error=/faststorage/project/imanol-pib/people/imanolhpc/gpn/human_sm_train_%j.err

# Conda
source ~/.bashrc
conda activate gpn-gpu

# Working Directory
cd /faststorage/project/imanol-pib/people/imanolhpc/gpn

# NCCL Config 4 GPUs
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=PHB
export PYTHONUNBUFFERED=1


export MASTER_PORT=$(shuf -i 29500-32000 -n 1)

# Launch
torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    -m gpn.cli ss train recipes/gpn_training/human_sm.yaml

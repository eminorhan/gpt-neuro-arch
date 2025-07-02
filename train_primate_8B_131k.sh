#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=21
##SBATCH --exclude=arch[09-42]
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --job-name=train_primate_8b_131k
#SBATCH --output=train_primate_8b_131k_%A_%a.out
#SBATCH --array=0

# activate venv
# source /lustre/gale/stf218/scratch/emin/myvenv/bin/activate
eval "$(/lustre/gale/stf218/scratch/emin/container/miniconda3/bin/conda shell.bash hook)"

# set misc env vars
export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=3   # can improve performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1       # on large systems, this nccl setting has been found to improve performance
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/lustre/gale/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/gale/stf218/scratch/emin/huggingface"
export TRITON_CACHE_DIR="/lustre/gale/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/gale/stf218/scratch/emin/pytorch_kernel_cache"
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=4

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/primate_8b_131k.toml"}

srun torchrun --nnodes $SLURM_NNODES --nproc_per_node 4 --max_restarts 9 --node_rank $SLURM_NODEID --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" ./train.py --job.config_file ${CONFIG_FILE}

echo "Done"
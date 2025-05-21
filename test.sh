#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=36
##SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --job-name=test
#SBATCH --output=test_%A_%a.out
#SBATCH --array=0
##SBATCH --qos=debug

# set proxy server to enable communication with outside
# export all_proxy=socks://proxy.ccs.ornl.gov:3128/
# export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
# export http_proxy=http://proxy.ccs.ornl.gov:3128/
# export https_proxy=http://proxy.ccs.ornl.gov:3128/
# export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

source /lustre/gale/stf218/scratch/emin/myvenv/bin/activate

# set misc env vars
export LOGLEVEL=INFO
export NCCL_NET_GDR_LEVEL=3   # can improve performance, but remove this setting if you encounter a hang/crash.
#export NCCL_ALGO=TREE         # may see performance difference with either setting. (should not need to use this, but can try)
export NCCL_CROSS_NIC=1       # on large systems, this nccl setting has been found to improve performance
# export NCCL_SOCKET_IFNAME=hsn0
# export GLOO_SOCKET_IFNAME=hsn0
# export NCCL_IB_TIMEOUT=31
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCHELASTIC_ENABLE_FILE_TIMER=1
# export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NCCL variables
# export NCCL_IB_DISABLE=1
# export NCCL_BUFFSIZE=2097152
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_NVLS_ENABLE=0

#export NCCL_NET="IB"
#export NCCL_IB_DISABLE=0
# module load libfabric/1.22.0
# export LD_LIBRARY_PATH=/lustre/gale/stf218/scratch/emin/aws-ofi-nccl/lib:$LD_LIBRARY_PATH  # enable aws-ofi-nccl
export HF_HOME="/lustre/gale/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/gale/stf218/scratch/emin/huggingface"
export TRITON_CACHE_DIR="/lustre/gale/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/gale/stf218/scratch/emin/pytorch_kernel_cache"
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=4

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/test.toml"}

srun torchrun --nnodes $SLURM_NNODES --nproc_per_node 4 --max_restarts 9 --node_rank $SLURM_NODEID --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" ./train.py --job.config_file ${CONFIG_FILE}

echo "Done"
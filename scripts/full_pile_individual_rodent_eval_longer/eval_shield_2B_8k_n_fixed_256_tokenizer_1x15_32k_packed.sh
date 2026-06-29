#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=38                  # Request all 38 nodes at once
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --job-name=eval_full_pile_shield_packed
#SBATCH --output=eval_full_pile_shield_packed_%j.out   # %j gets replaced with the Job ID

# activate venv
source /lustre/blizzard/stf218/scratch/emin/blizzardvenv/bin/activate

# set misc env vars
export LD_LIBRARY_PATH=/lustre/blizzard/stf218/scratch/emin/aws-ofi-nccl-1.19.0/lib:$LD_LIBRARY_PATH
export NCCL_NET=ofi
export FI_PROVIDER=cxi
export LOGLEVEL=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export HF_HOME="/lustre/blizzard/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/blizzard/stf218/scratch/emin/huggingface"
export TRITON_CACHE_DIR="/lustre/blizzard/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/blizzard/stf218/scratch/emin/pytorch_kernel_cache"
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=4

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/full_pile/rodent_2B_8k_n_fixed_256_tokenizer_1x15_32k_longer.toml"}
CHECKPOINT_DIR="./outputs/rodent_2B_8k_n_fixed_256_tokenizer_1x15_32k_longer/checkpoint"

# Get all checkpoints
CHECKPOINTS=($(ls -d ${CHECKPOINT_DIR}/step-* | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found."
    exit 1
fi

# Expand the list of allocated nodes into a bash array
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODES_PER_JOB=2
MAX_CONCURRENT_JOBS=$(( ${#NODES[@]} / NODES_PER_JOB ))

echo "Total nodes allocated: ${#NODES[@]}"
echo "Max concurrent 2-node jobs: $MAX_CONCURRENT_JOBS"

# Loop through all available checkpoints
for i in "${!CHECKPOINTS[@]}"; do
    CKPT_PATH=${CHECKPOINTS[$i]}
    
    # Calculate which subset (slot) of nodes this job will use
    SLOT=$(( i % MAX_CONCURRENT_JOBS ))
    
    # Identify the specific 2 nodes for this subset
    NODE1=${NODES[$((SLOT * NODES_PER_JOB))]}
    NODE2=${NODES[$((SLOT * NODES_PER_JOB + 1))]}
    SUB_NODELIST="$NODE1,$NODE2"
    
    # Set unique rendezvous parameters for this specific torchrun job
    MASTER_ADDR=$NODE1
    MASTER_PORT=$(( 3442 + i )) # Increment port to guarantee no collisions
    
    echo "Starting shield eval for $CKPT_PATH on nodes $SUB_NODELIST (Master: $MASTER_ADDR:$MASTER_PORT)"
    
    # Launch srun in the background for this node subset
    srun --nodelist=${SUB_NODELIST} \
         --nodes=${NODES_PER_JOB} \
         --ntasks=${NODES_PER_JOB} \
         --ntasks-per-node=1 \
         --exclusive \
         bash -c "torchrun --nnodes ${NODES_PER_JOB} \
                  --nproc_per_node 4 \
                  --max_restarts 1 \
                  --node_rank \$SLURM_NODEID \
                  --rdzv_id 101 \
                  --rdzv_backend c10d \
                  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
                  ./evaluate.py --config ${CONFIG_FILE} \
                  --ckpt ${CKPT_PATH} \
                  --eval_steps 1000 \
                  --eval_dirname \"eval-shield\" \
                  --source_dataset \"shield\"" \
         > "eval_shield_log_${i}.out" 2>&1 &
    
    # If we've filled all available slots, wait for this batch to finish before continuing
    if [[ $(( (i + 1) % MAX_CONCURRENT_JOBS )) == 0 ]]; then
        echo "Filled $MAX_CONCURRENT_JOBS slots. Waiting for current batch to complete..."
        wait
    fi
done

# Wait for any remaining background jobs in the final partial batch
wait
echo "All shield evaluations complete."
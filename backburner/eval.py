# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import time
import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims

# Import directly from your dataset.py
from dataset import build_data_loader

# --- CONFIGURATION ---
EVAL_DATASET = "eminorhan/neural-pile-primate" 
# ---------------------

@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting Evaluation Job: {job_config.job.description}")
    logger.info(f"Target Dataset: {EVAL_DATASET}")

    # 1. Setup Distributed Environment
    utils.set_determinism(job_config.training.seed)
    world_size = int(os.environ['WORLD_SIZE'])
    
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    # 2. Build Model
    model_name = job_config.model.name
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = job_config.training.vocab_size
    model_config.max_seq_len = job_config.training.seq_len
    model_config.rope_theta = job_config.model.rope_theta

    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # 3. Parallelize & Load Checkpoint
    models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)
    model.to_empty(device="cuda")
    model.eval()

    train_state = TrainState()
    
    # We pass None for dataloader, and dummy [None] lists for optim/schedulers
    # to satisfy the CheckpointManager assertions regarding list lengths.
    checkpoint = CheckpointManager(
        dataloader=None,
        model_parts=[model],
        optimizers=[None], 
        lr_schedulers=[None],
        states={"train_state": train_state},
        job_config=job_config,
    )

    # Clean up None entries from the states dict.
    # dcp.load() will error if it tries to load checkpoint data into a None target.
    keys_to_remove = ["dataloader", "optimizer", "lr_scheduler"]
    for key in keys_to_remove:
        if checkpoint.states.get(key) is None:
            checkpoint.states.pop(key, None)

    if not checkpoint.load():
        logger.warning("No checkpoint loaded! Evaluating random weights.")
    else:
        logger.info(f"Loaded checkpoint from step {train_state.step}")

    # 4. Define Loss
    token_weights = torch.load(job_config.training.token_weights) if job_config.training.token_weights else None
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1), labels.flatten(0, 1), 
            weight=token_weights.to(pred.device) if token_weights is not None else None,
            reduction='sum'
        )

    # 5. Build Eval Loader
    loader = build_data_loader(
        dataset_name=EVAL_DATASET,
        dataset_path=job_config.training.dataset_path,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        vocab_size=job_config.training.vocab_size,
        world_size=dp_degree,
        rank=dp_rank,
        infinite=False, # Single pass
        tokenizer_path=job_config.model.tokenizer_path,
        n_fixed=job_config.training.n_fixed,
        split="test"    # Test split
    )

    # 6. Run Evaluation
    total_loss = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)
    
    logger.info("Starting inference loop...")
    start_time = time.time()
    
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(loader):
            input_ids, labels = input_ids.cuda(), labels.cuda()
            pred = model(input_ids)
            
            total_loss += loss_fn(pred, labels)
            total_tokens += labels.numel()
            
            if i % 10 == 0 and utils.get_rank() == 0:
                print(f"Processed batch {i}...", end='\r')

    # Global aggregation
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tokens, op=torch.distributed.ReduceOp.SUM)
    
    if utils.get_rank() == 0:
        duration = time.time() - start_time
        avg_loss = (total_loss / total_tokens).item()
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info("-" * 50)
        logger.info(f"RESULTS for {EVAL_DATASET}")
        logger.info(f"  - Total Tokens: {int(total_tokens.item())}")
        logger.info(f"  - Avg Loss:     {avg_loss:.4f}")
        logger.info(f"  - Perplexity:   {perplexity:.4f}")
        logger.info(f"  - Time:         {duration:.2f}s")
        logger.info("-" * 50)

    logger.info("Evaluation Complete")

if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
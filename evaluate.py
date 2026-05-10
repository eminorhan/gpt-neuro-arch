# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
import re

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.elastic.multiprocessing.errors import record

# support running w/o installing as package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_data_loader
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import models_parallelize_fns, models_pipelining_fns, ParallelDims


def get_eval_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))
            yield

    return context


@record
def main(config_path: str, checkpoint_path: str, eval_steps: int):
    job_config = JobConfig()
    job_config.parse_args([f"--job.config_file={config_path}"])
    job_config._validate_config()

    init_logger()
    logger.info(f"Starting evaluation for job: {job_config.job.description}")

    # set determinism, use seed == None to skip deterministic training
    utils.set_determinism(job_config.training.seed)

    # init distributed
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)

    # initialize GPU memory monitor
    gpu_memory_monitor = build_gpu_memory_monitor()
    
    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # build dataloader
    data_loader = build_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        job_config.training.batch_size,
        job_config.training.seq_len,
        job_config.training.vocab_size,
        dp_degree,
        dp_rank,
        infinite=False,
        tokenizer_path=job_config.model.tokenizer_path,
        n_fixed=job_config.training.n_fixed,
        split="test"
    )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = job_config.training.vocab_size
    model_config.max_seq_len = job_config.training.seq_len
    model_config.rope_theta = job_config.model.rope_theta

    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    float8_handler = Float8Handler(job_config, parallel_dims)
    float8_handler.convert_to_float8_training(model)

    token_weights = torch.load(job_config.training.token_weights) if job_config.training.token_weights else None
    if torch.distributed.get_rank() == 0:
        logger.info(f"Using cross-entropy with token weights from {job_config.training.token_weights}: {token_weights}")
        
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1), weight=token_weights.to(pred.device) if token_weights is not None else None)

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        pp_schedule, model_parts = models_pipelining_fns[model_name](model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn)
        for m in model_parts:
            models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
            m.to_empty(device="cuda")
            m.init_weights()
            m.eval()
    else:
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)
        model.to_empty(device="cuda")
        model.init_weights()
        model.eval()
        model_parts = [model]

    # use CheckpointManager to load the checkpoint
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)
    train_state = TrainState()

    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    # Override states to ONLY load the model. This prevents OOM by avoiding optimizer state loading,
    # and prevents crashes when resharding states across different node counts.
    checkpoint.states = {"model": checkpoint.states["model"]}

    match = re.search(r"step-(\d+)", checkpoint_path)
    step = int(match.group(1)) if match else -1
    if step != -1:
        checkpoint.folder = os.path.dirname(os.path.normpath(checkpoint_path))

    logger.info(f"Loading checkpoint from: {checkpoint_path} (step {step})")
    if not checkpoint.load(step=step):
        logger.error(f"Failed to load checkpoint from {checkpoint_path}")
        sys.exit(1)
    logger.info("Checkpoint loaded.")

    eval_context = get_eval_context(parallel_dims.loss_parallel_enabled, job_config.experimental.enable_compiled_autograd)

    total_sum_loss = 0.0
    total_valid_batches = 0.0
    num_steps = 0

    logger.info("Evaluation starts.")
    data_iterator = iter(data_loader)
    
    last_batch = None

    with torch.no_grad():
        while True:
            if eval_steps > 0 and num_steps >= eval_steps:
                break
            
            try:
                batch = next(data_iterator)
                has_data_local = True
                last_batch = batch
            except StopIteration:
                has_data_local = False
                batch = last_batch

            if batch is None:
                # If a rank has no data from the start, create a dummy batch to avoid hanging the process group
                dummy_input = torch.zeros(job_config.training.batch_size, job_config.training.seq_len, dtype=torch.long)
                dummy_label = torch.zeros(job_config.training.batch_size, job_config.training.seq_len, dtype=torch.long)
                batch = (dummy_input, dummy_label)
                last_batch = batch

            has_data_tensor = torch.tensor(1 if has_data_local else 0, device="cuda")

            if parallel_dims.dp_enabled:
                # Synchronize data availability. If ALL ranks are out of data, MAX makes it 0 for everyone
                torch.distributed.all_reduce(has_data_tensor, op=torch.distributed.ReduceOp.MAX, group=dp_mesh.get_group("dp"))

            if has_data_tensor.item() == 0:
                break

            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            if parallel_dims.pp_enabled:
                is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                with eval_context():
                    if pp_mesh.get_local_rank() == 0:
                        pp_schedule.step(input_ids)
                    elif is_last_stage:
                        losses = []
                        pp_schedule.step(target=labels, losses=losses)
                    else:
                        pp_schedule.step()

                loss = torch.mean(torch.stack(losses)) if is_last_stage else torch.Tensor([-1.0])
            else:
                is_last_stage = True
                with eval_context():
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    del pred

            if is_last_stage:
                local_loss = loss.item() if has_data_local else 0.0
                local_count = 1.0 if has_data_local else 0.0

                if parallel_dims.dp_enabled:
                    loss_tensor = torch.tensor([local_loss, local_count], device="cuda")
                    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM, group=dp_mesh.get_group("dp"))
                    step_loss_sum = loss_tensor[0].item()
                    step_count = loss_tensor[1].item()
                else:
                    step_loss_sum = local_loss
                    step_count = local_count

                total_sum_loss += step_loss_sum
                total_valid_batches += step_count
                num_steps += 1

                if num_steps == 1 or num_steps % job_config.metrics.log_freq == 0:
                    step_avg_loss = step_loss_sum / step_count if step_count > 0 else 0.0
                    logger.info(f"Eval step {num_steps}: loss {step_avg_loss:.4f} (valid batches across DP: {int(step_count)})")

    if is_last_stage:
        if total_valid_batches > 0:
            avg_eval_loss = total_sum_loss / total_valid_batches
        else:
            avg_eval_loss = float('nan')

        logger.info(f"Evaluation completed. Average loss: {avg_eval_loss:.4f} over {int(total_valid_batches)} total valid batches.")

        if dp_rank == 0:
            eval_dir = os.path.join(job_config.job.dump_folder, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            ckpt_name = os.path.basename(os.path.normpath(checkpoint_path))
            result_file = os.path.join(eval_dir, f"eval_result_{ckpt_name}.json")
            result = {
                "checkpoint": checkpoint_path,
                "avg_cross_entropy_loss": avg_eval_loss,
                "num_batches": int(total_valid_batches),
                "num_steps": num_steps,
                "seq_len": job_config.training.seq_len,
                "batch_size": job_config.training.batch_size
            }
            with open(result_file, "w") as f:
                json.dump(result, f, indent=4)
            logger.info(f"Evaluation results saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pretrained checkpoint")
    parser.add_argument("--config", type=str, required=True, help="TOML config file path")
    parser.add_argument("--ckpt", type=str, required=True, help="DCP checkpoint path to evaluate")
    parser.add_argument("--eval_steps", type=int, default=-1, help="Max number of evaluation steps (optional)")
    args = parser.parse_args()

    main(args.config, args.ckpt, args.eval_steps)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
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

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.elastic.multiprocessing.errors import record

# support running w/o installing as package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from torchtitan import utils
from torchtitan.checkpoint import ModelWrapper
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_data_loader
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.models import model_name_to_cls, models_config
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

    # load checkpoint
    state_dict = {"model": ModelWrapper(model_parts)}

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info("Checkpoint loaded.")

    eval_context = get_eval_context(parallel_dims.loss_parallel_enabled, job_config.experimental.enable_compiled_autograd)

    total_loss = 0.0
    num_batches = 0

    logger.info("Evaluation starts.")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if eval_steps > 0 and i >= eval_steps:
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
                if parallel_dims.dp_enabled:
                    global_avg_loss = utils.dist_mean(loss.item(), dp_mesh)
                else:
                    global_avg_loss = loss.item()

                total_loss += global_avg_loss
                num_batches += 1

                if i == 0 or (i + 1) % job_config.metrics.log_freq == 0:
                    logger.info(f"Eval step {i + 1}: loss {global_avg_loss:.4f}")

    if is_last_stage:
        if num_batches > 0:
            avg_eval_loss = total_loss / num_batches
        else:
            avg_eval_loss = float('nan')

        logger.info(f"Evaluation completed. Average loss: {avg_eval_loss:.4f}")

        if dp_rank == 0:
            eval_dir = os.path.join(job_config.job.dump_folder, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            ckpt_name = os.path.basename(os.path.normpath(checkpoint_path))
            result_file = os.path.join(eval_dir, f"eval_result_{ckpt_name}.json")
            result = {
                "checkpoint": checkpoint_path,
                "avg_cross_entropy_loss": avg_eval_loss,
                "num_batches": num_batches,
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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
import time
from pathlib import Path

from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import numpy as np
from datasets import load_dataset

from torch.distributed import DeviceMesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.config_manager import JobConfig
from torchtitan.parallelisms import ParallelDims
from torchtitan import utils as dist_utils

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from torchtitan.generation import generate


def apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


@record
def test_generate(
    config_path: str,
    checkpoint_path: str,
    data_idx: int,
    ctx_t: int,
    gen_t: int,
    *,
    temperature: float = 1.0,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
):
    init_logger()

    # Load configuration from toml file
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    gpu_memory_monitor = build_gpu_memory_monitor()

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # model setup
    model_name = config.model.name
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][config.model.flavor]
    model_config.norm_type = config.model.norm_type
    model_config.vocab_size = config.training.vocab_size
    model_config.max_seq_len = config.training.seq_len

    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_model_args(model_config)

    world_mesh = None
    # Init distributed env
    if world_size > 1:
        dist_utils.init_distributed(config)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            tp=8,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        # Build world mesh for parallelism
        world_mesh = parallel_dims.build_mesh(device_type="cuda")

        # apply_tp (with Sequence Parallel) on unevenly sharded
        # sequences would require https://github.com/pytorch/torchtitan/pull/686
        apply_tp_minus_sp(model, world_mesh["tp"])

    dist_utils.set_determinism(seed)

    # materalize model
    model.to_empty(device="cuda")
    model.eval()

    state_dict = {"model": model.state_dict()}

    # load ckpt
    begin = time.monotonic()
    logger.info(f"Loading ckpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Finished loading ckpt in {time.monotonic() - begin:.2f} seconds.")

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(f"GPU memory usage for model: {gpu_mem_stats.max_reserved_gib:.2f}GiB ({gpu_mem_stats.max_reserved_pct:.2f}%)")

    # set up input
    ds = load_dataset("eminorhan/neural-bench-primate", split="train")
    logger.info(f"Test dataset loaded (size: {len(ds)})")

    data_row = ds[data_idx]
    source_dataset = data_row["source_dataset"]
    sample = np.array(data_row["spike_counts"])
    logger.info(f"Sample loaded (shape: {sample.shape})")
    logger.info(f"Sample source dataset: {source_dataset})")

    n_neurons = sample.shape[0]
    bos_token = model_config.vocab_size - 1
    max_new_tokens = (n_neurons + 1) * gen_t  # total number of tokens to be generated (+1 for bos)

    # append bos token
    sample = np.concatenate((np.full((1, sample.shape[1]), bos_token), sample), axis=0)

    prompt = sample[:, :ctx_t]  # prompt
    prompt = prompt.T.flatten().tolist()

    gt = sample[:, :(ctx_t+gen_t)]  # ground truth
    gt = gt.T.flatten().tolist()

    input_ids = torch.tensor(prompt, dtype=torch.long).view(1, -1).repeat(batch_size, 1).to("cuda")

    gpu_memory_monitor.reset_peak_stats()

    # generate
    t0 = time.monotonic()
    responses = generate(
        model,
        input_ids,
        n_neurons,
        bos_token,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
    )
    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Post process
    B, T = responses.size()  # B: batch_size, T: total seq length
    input_n_tokens = input_ids.size(1)
    generated_n_tokens = T - input_n_tokens  # == max_new_tokens

    if local_rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            _data = {
                "response_idx": i,
                "input_tok": inp_tok,
                "output_tok": out_tok,
            }
            output_data["responses"].append(_data)

            logger.info(f"\n{inp_tok} - {out_tok}\n")
            np.savez(f"rodent_test_sample_{data_idx}_{ctx_t}_{gen_t}.npz", prompt=inp_tok, gen=out_tok, gt=gt)

        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
        output_data["metadata"] = {
            "generated_n_tokens": generated_n_tokens,
            "input_n_tokens": input_n_tokens,
            "generation_time_sec": elapsed_sec,
            "tokens_per_sec": (B * T) / elapsed_sec,
            "batch_size": B,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
            "memory/max_active(%)": gpu_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
            "memory/num_ooms": gpu_mem_stats.num_ooms,
            "world_size": world_size,
            "torch_version": torch.__version__,
        }

        if args.out:
            print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument("--config", type=str, required=True, help="TOML config file path (required)")
    parser.add_argument("--ckpt", type=str, required=True, help="DCP checkpoint path to load (required)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Default is 1.0")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to run in batch")
    parser.add_argument("--top_k", type=int, help="Prune to select from top_k probabilities. Optional")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--data_idx", type=int, default=2, help="Idx of data prompt")
    parser.add_argument("--ctx_t", type=int, default=10, help="Duration of prompt context (time bins)")
    parser.add_argument("--gen_t", type=int, default=10, help="Duration of generated sample (time bins)")
    parser.add_argument("--out", action="store_true", default=False, help="If specified, prints the report to stdout. Defaults to no output.")

    args = parser.parse_args()
    print(args)

    test_generate(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        data_idx=args.data_idx,
        ctx_t = args.ctx_t,
        gen_t = args.gen_t,
        temperature=args.temperature,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

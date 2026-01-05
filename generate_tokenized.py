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
import pickle
from pathlib import Path

from typing import Optional, Tuple, Dict, Any

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

import matplotlib.pyplot as plt

def load_tokenizer(path: str) -> Dict[str, Any]:
    """
    Placeholder for loading the tokenizer. 
    Assumes the tokenizer file is a pickle or json containing the 'index_map', 'patch_size', etc.
    """
    # NOTE: Implementation omitted as per instructions. 
    # In a real scenario, this would load the dictionary from disk.
    with open(path, 'rb') as f:
         return pickle.load(f)

def get_patches_column_major(data_array: np.ndarray, patch_size: Tuple[int, int]) -> Tuple[np.ndarray, int]:
    """
    Pads a 2D array and extracts patches in column-major order.
    """
    if data_array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional, but got {data_array.ndim} dimensions.")
        
    p0, p1 = patch_size
    if not (p0 > 0 and p1 > 0):
        raise ValueError(f"Patch dimensions must be positive, but got ({p0}, {p1}).")

    n, t = data_array.shape

    # 1. Calculate padding
    pad_n = (p0 - (n % p0)) % p0
    pad_t = (p1 - (t % p1)) % p1

    # 2. Apply padding if needed
    if pad_n > 0 or pad_t > 0:
        padded_array = np.pad(
            data_array,
            ((0, pad_n), (0, pad_t)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_array = data_array
    
    # Get the new, padded dimensions
    N, T = padded_array.shape
    
    # Calculate the number of patches along each dimension
    num_patches_n = N // p0
    num_patches_t = T // p1

    # Reshape into a 4D array: (num_patches_n, p0, num_patches_t, p1)
    reshaped = padded_array.reshape(num_patches_n, p0, num_patches_t, p1)

    # Transpose to: (num_patches_t, num_patches_n, p0, p1)
    transposed = reshaped.transpose(2, 0, 1, 3)
    
    # Reshape to flatten patch indices
    patches = transposed.reshape(-1, p0, p1)

    return patches, num_patches_n

def standardize_neuron_dim(sample, n_fixed):
    """
    Adjusts the neuron dimension (rows) of the sample to exactly n_fixed.
    """
    n, t = sample.shape
    
    if n == n_fixed:
        return sample
    
    # Case 1: n < n_fixed (Pad with contiguous rows from the start)
    if n < n_fixed:
        repeats = (n_fixed // n) + 1
        extended_sample = np.tile(sample, (repeats, 1))
        return extended_sample[:n_fixed, :]
        
    # Case 2: n > n_fixed (Sample a random contiguous block)
    else:
        max_start_idx = n - n_fixed
        start_idx = np.random.randint(0, max_start_idx + 1)
        return sample[start_idx : start_idx + n_fixed, :]

def plot_data(data, filename='samples.jpeg'):
    """Plots data onto an Axes."""
    plt.figure(figsize=(10, 6))
    plt.imshow(data, interpolation='nearest', aspect='auto', cmap='gray_r')
    plt.xlim([-1, data.shape[-1] + 1])
    plt.ylim([-1, data.shape[0] + 1])
    plt.xticks([0, data.shape[-1] - 1])
    plt.yticks([0, data.shape[0] - 1])
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

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
    tokenizer_path: str,
    data_idx: int,
    ctx_t: int,
    gen_t: int,
    n_fixed_neurons: int,
    *,
    temperature: float = 1.0,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
):
    init_logger()

    # Load configuration
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    gpu_memory_monitor = build_gpu_memory_monitor()
    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # Load Tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    # Create inverse map for detokenization (Token Index -> Bytes)
    inv_index_map = {v: k for k, v in tokenizer["index_map"].items()}
    patch_size = tokenizer["patch_size"]
    # We assume vocab_size is provided in config or derived. 
    # If using tokenizer vocab size:
    # vocab_size = len(tokenizer["index_map"]) + N_SPECIAL_TOKENS
    
    # Model setup
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
    if world_size > 1:
        dist_utils.init_distributed(config)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            tp=4,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        apply_tp_minus_sp(model, world_mesh["tp"])

    dist_utils.set_determinism(seed)
    model.to_empty(device="cuda")
    model.eval()

    state_dict = {"model": model.state_dict()}
    begin = time.monotonic()
    logger.info(f"Loading ckpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Finished loading ckpt in {time.monotonic() - begin:.2f} seconds.")

    # --- DATA PREPARATION ---
    ds = load_dataset("eminorhan/neural-pile-primate", split="train")
    data_row = ds[data_idx]
    source_dataset = data_row["source_dataset"]
    logger.info(f"Sample source dataset: {source_dataset}")

    raw_sample = np.array(data_row["spike_counts"], dtype=np.uint8)
    
    # 1. Standardize Neurons
    sample = standardize_neuron_dim(raw_sample, n_fixed_neurons)
    logger.info(f"Sample standardized to shape: {sample.shape}")

    # 2. Tokenize
    patchified_sample, num_patches_n = get_patches_column_major(sample, patch_size)
    
    # Map to tokens
    token_sequence = [tokenizer["index_map"].get(patch.tobytes(), 0) for patch in patchified_sample]
    
    token_arr = np.array(token_sequence)
    
    # Reshape to (Time_Steps, Spatial_Patches_Per_Step)
    # The number of patches per time step (column) is num_patches_n
    token_arr = token_arr.reshape(-1, num_patches_n)

    # Insert Separator Column
    # The vocab size in config must accommodate the separator. 
    # Usually sep_token is the last index.
    sep_token = model_config.vocab_size - 1 
    sep_col = np.full((token_arr.shape[0], 1), sep_token, dtype=token_arr.dtype)
    
    # Stack: [SEP, Token1, Token2, ...] -> Shape (Total_Time_Steps, Patches_Per_Step + 1)
    token_arr = np.hstack((sep_col, token_arr))
    
    logger.info(f"Tokenized array shape: {token_arr.shape}")

    # Define context and generation length (in Time Steps)
    # We slice the token_arr by rows (time)
    prompt_tokens = token_arr[:ctx_t, :]
    gt_tokens = token_arr[:(ctx_t + gen_t), :]
    
    # Flatten for model input
    prompt_flat = prompt_tokens.flatten().tolist()
    gt_flat = gt_tokens.flatten().tolist()

    input_ids = torch.tensor(prompt_flat, dtype=torch.long).view(1, -1).repeat(batch_size, 1).to("cuda")

    # For generation, we calculate how many *tokens* we need to generate
    # Each time step contributes (num_patches_n + 1) tokens
    tokens_per_step = num_patches_n + 1
    max_new_tokens = gen_t * tokens_per_step

    gpu_memory_monitor.reset_peak_stats()

    t0 = time.monotonic()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Pass dummy n_neurons if generate() doesn't strictly use it for logic other than logging/reshaping
        # Since we are feeding tokens, we pass the tokens.
        responses = generate(
            model,
            input_ids,
            tokens_per_step - 1, # passing tokens_per_step-1 as "n_neurons" proxy if used for shaping
            sep_token, # acting as bos_token
            logger,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            seed=seed,
        )
    elapsed_sec = time.monotonic() - t0

    B, T_total_tokens = responses.size()
    input_n_tokens = input_ids.size(1)
    
    if local_rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()
            full_tok = tokens.tolist()

            _data = {
                "response_idx": i,
                "input_tok": inp_tok,
                "output_tok": out_tok,
            }
            output_data["responses"].append(_data)
            logger.info(f"\n{inp_tok} - {out_tok}\n")

            # --- DETOKENIZATION & PLOTTING ---
            
            # 1. Reshape flat tokens back to (Total_Time, Patches_Per_Step + 1)
            # T_total_tokens should be divisible by tokens_per_step
            n_rows = len(full_tok) // tokens_per_step
            reconstructed_arr = np.array(full_tok).reshape(n_rows, tokens_per_step)
            
            # 2. Remove Separator (first column)
            # Shape becomes (Total_Time, Patches_Per_Step) == (T, num_patches_n)
            patch_indices = reconstructed_arr[:, 1:] 
            
            # 3. Flatten back to list of patches for mapping
            # (T, num_patches_n) -> flatten -> Iterate
            flat_indices = patch_indices.flatten()
            
            reconstructed_patches = []
            for idx in flat_indices:
                # Map token index -> bytes -> numpy array
                if idx in inv_index_map:
                    b_data = inv_index_map[idx]
                    patch = np.frombuffer(b_data, dtype=sample.dtype).reshape(patch_size)
                    logger.info(f"token idx: {idx} - patch: {patch}")
                else:
                    # Fallback for unknown tokens (shouldn't happen with valid sampling)
                    patch = np.zeros(patch_size, dtype=sample.dtype)
                    logger.info(f"Fallback to zeros...")
                reconstructed_patches.append(patch)
            
            reconstructed_patches = np.array(reconstructed_patches)
            
            # 4. Stitch patches back to 2D Array
            # The patches were created via:
            # reshaped = padded_array.reshape(num_patches_n, p0, num_patches_t, p1)
            # transposed = reshaped.transpose(2, 0, 1, 3) --> (num_patches_t, num_patches_n, p0, p1)
            
            # We currently have patches in the order of 'transposed' flattened.
            # reconstructed_patches shape: (Total_Patches, p0, p1)
            
            num_patches_t = n_rows # Since we sliced by time rows earlier
            p0, p1 = patch_size
            
            # Reshape to (num_patches_t, num_patches_n, p0, p1)
            reshaped_patches = reconstructed_patches.reshape(num_patches_t, num_patches_n, p0, p1)
            
            # Inverse Transpose: We want (num_patches_n, p0, num_patches_t, p1)
            # Current: (dim0, dim1, dim2, dim3) -> Target: (dim1, dim2, dim0, dim3)
            # Map: 0->2, 1->0, 2->1, 3->3
            # Inverse map: 1, 2, 0, 3
            original_order = reshaped_patches.transpose(1, 2, 0, 3)
            
            # Fuse dimensions to get (N, T)
            # (num_patches_n, p0, num_patches_t, p1) -> (num_patches_n * p0, num_patches_t * p1)
            full_height = num_patches_n * p0
            full_width = num_patches_t * p1
            
            final_image = original_order.reshape(full_height, full_width)
            
            # Determine split point for plotting
            prompt_width = ctx_t * p1 # Approximate pixel width of prompt
            
            # Plot
            plot_name = f"sample_{data_idx}_ctx{ctx_t}_gen{gen_t}_{i}.jpeg"
            logger.info(f"Saving plot to {plot_name}")
            plot_data(final_image, filename=plot_name)

            # # Save
            # np.savez(f"sample_{data_idx}_{ctx_t}_{gen_t}.npz", prompt_tok=inp_tok, gen_tok=out_tok, reconstructed_data=final_image)

        if args.out:
            print(json.dumps(output_data, indent=4, default=str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument("--config", type=str, required=True, help="TOML config file path")
    parser.add_argument("--ckpt", type=str, required=True, help="DCP checkpoint path")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer pickle/json")
    parser.add_argument("--n_fixed_neurons", type=int, default=256, help="Standardize input to this many neurons")
    
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, help="Prune top_k")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_idx", type=int, default=19)
    parser.add_argument("--ctx_t", type=int, default=30, help="Context time steps")
    parser.add_argument("--gen_t", type=int, default=1, help="Generation time steps")
    parser.add_argument("--out", action="store_true", default=False)

    args = parser.parse_args()

    test_generate(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        tokenizer_path=args.tokenizer_path,
        data_idx=args.data_idx,
        ctx_t=args.ctx_t,
        gen_t=args.gen_t,
        n_fixed_neurons=args.n_fixed_neurons,
        temperature=args.temperature,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
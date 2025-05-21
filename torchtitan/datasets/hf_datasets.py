# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import numpy as np
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.logging import logger

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

# map from dataset name to a local directory, or a dataset repository on the HF hub
_supported_datasets = {
    "rodent": "eminorhan/neural-pile-rodent",
    "primate": "eminorhan/neural-pile-primate",
}

class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        seq_len: int = 131072,
        vocab_size: int = 256,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = True,
    ) -> None:
        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(f"Dataset {dataset_name} is not tested or verfied. Recommended datasets are: {list(_supported_datasets.keys())}")
            else:
                raise ValueError(f"Dataset {dataset_name} is not supported. Supported datasets are: {list(_supported_datasets.keys())}")

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")
        ds = load_dataset(dataset_path, split="train")

        # NOTE: datasets are pre-shuffled
        self._data = split_dataset_by_node(ds, rank, world_size)
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.infinite = infinite

        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample = np.array(sample['spike_counts'])
                # print(f"Sample dtype-1 / shape: {sample.dtype} / {sample.shape}")
                sample = np.concatenate((np.full((1, sample.shape[1]), self.vocab_size-1), sample), axis=0)
                # print(f"Sample dtype-2 / shape: {sample.dtype} / {sample.shape}") 
                sample = sample.T.flatten().tolist()  
                self._all_tokens.extend(sample)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    # print(f"_all_tokens length: {len(self._all_tokens)}")
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def _get_data_iter(self):
        # as skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)

        return it
    
    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}")
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDataset(dataset_name, dataset_path, seq_len, vocab_size, world_size, rank, infinite)

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
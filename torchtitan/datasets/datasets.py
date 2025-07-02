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
                sample = np.concatenate((np.full((1, sample.shape[1]), self.vocab_size-1), sample), axis=0)
                sample = sample.T.flatten().tolist()
                self._all_tokens.extend(sample)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
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


class SyntheticDataset(IterableDataset, Stateful):
    """PyTorch IterableDataset for generating synthetic data on-the-fly.

    This dataset generates random matrices, simulating a stream of data for training.
    It is stateful and supports checkpointing to ensure reproducibility in a
    distributed environment.

    Args:
        seq_len (int): max sequence length
        vocab_size (int): vocabulary size
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset
    """

    def __init__(
        self,
        seq_len: int = 131072,
        vocab_size: int = 256,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.infinite = infinite
        self.rank = rank

        # seed the rng for this process to ensure different data per rank
        # adding rank to a random seed ensures that each process starts with a
        # unique, non-overlapping sequence of random numbers
        np.random.seed(rank + np.random.randint(0, 2**32 - 1))

        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []
        self._rng_state = np.random.get_state()

    def _generate_sample(self) -> np.ndarray:
        """Generates a single synthetic data sample."""
        rows = np.random.randint(10, 1000)
        cols = np.random.randint(100, 2000)
        sample = np.zeros((rows, cols))
        num_active_rows = int(rows * 0.1)
        random_indices = np.random.choice(rows, size=num_active_rows, replace=False)
        sample[random_indices] = 1
        return sample

    def __iter__(self):
        # restore the RNG state at the beginning of iteration to ensure
        # that resuming from a checkpoint continues the same random sequence.
        np.random.set_state(self._rng_state)

        max_buffer_token_len = 1 + self.seq_len

        while True:
            sample = self._generate_sample()
            self._sample_idx += 1

            # process the sample similarly to HuggingFaceDataset
            sample = np.concatenate((np.full((1, sample.shape[1]), self.vocab_size - 1), sample), axis=0)
            sample = sample.T.flatten().tolist()
            self._all_tokens.extend(sample)

            # yield sequences from the buffer
            while len(self._all_tokens) >= max_buffer_token_len:
                x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                self._all_tokens = self._all_tokens[max_buffer_token_len:]
                input_seq = x[:-1]
                label = x[1:]
                yield input_seq, label

            # for synthetic data, 'infinite' is the natural mode
            # a hard stop is included for consistency if infinite=False
            if not self.infinite and self._sample_idx > 100000:
                 logger.warning(f"SyntheticDataset has reached its arbitrary limit of {self._sample_idx} samples.")
                 break

    def state_dict(self) -> Dict[str, Any]:
        # capture the current RNG state for checkpointing.
        self._rng_state = np.random.get_state()
        return {
            "token_buffer": self._all_tokens,
            "sample_idx": self._sample_idx,
            "rng_state": self._rng_state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]
        self._rng_state = state_dict["rng_state"]
        # rng state will be restored at the start of the next __iter__ call.


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, dataset: IterableDataset, batch_size: int):
        super().__init__(dataset, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # state being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}")
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    world_size,
    rank,
    infinite: bool = True,
) -> DPAwareDataLoader:
    """
    Builds a data loader for distributed training.

    This function can create a data loader for a Hugging Face dataset or a
    dataset with synthetically generated data.

    Args:
        dataset_name (str): The name of the dataset. Use "synthetic" to generate
            data on the fly. Otherwise, use a name from _supported_datasets.
        batch_size (int): The batch size for the data loader.
        seq_len (int): The sequence length of the samples.
        vocab_size (int): The vocabulary size.
        world_size (int): The total number of processes in the distributed group.
        rank (int): The rank of the current process.
        dataset_path (Optional[str]): Path to a local dataset. Required for
            unsupported Hugging Face datasets.
        infinite (bool): Whether the data loader should loop infinitely.

    Returns:
        DPAwareDataLoader: A configured stateful data loader for distributed training.
    """
    if dataset_name == "synthetic":
        logger.info(f"Using synthetic dataset for rank {rank}.")
        dataset = SyntheticDataset(
            seq_len=seq_len,
            vocab_size=vocab_size,
            world_size=world_size,
            rank=rank,
            infinite=infinite,
        )
    else:
        dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            seq_len=seq_len,
            vocab_size=vocab_size,
            world_size=world_size,
            rank=rank,
            infinite=infinite,
        )

    return DPAwareDataLoader(rank, dataset, batch_size=batch_size)
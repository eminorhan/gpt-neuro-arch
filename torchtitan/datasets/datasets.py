import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.logging import logger

from datasets import Dataset, load_dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node

# map from dataset name to a local directory, or a dataset repository on the HF hub
_supported_datasets = {
    "rodent": "eminorhan/neural-pile-rodent",
    "primate": "eminorhan/neural-pile-primate",
    "willett": "eminorhan/willett",
    "willett-churchland": ["eminorhan/willett", "eminorhan/churchland"],
    "willett-churchland-makin": ["eminorhan/willett", "eminorhan/churchland", "eminorhan/makin"],
    "card-willett-churchland-makin": ["eminorhan/card", "eminorhan/willett", "eminorhan/churchland", "eminorhan/makin"]
}

# some utility functions for tokenization
def get_patches_column_major(data_array: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Pads a 2D array and extracts patches in column-major order.

    Given a 2D (n, t) array and a patch size (p0, p1), this function:
    1. Pads the array with zeros so that n is divisible by p0 and t is divisible by p1.
    2. Extracts all (p0, p1) patches.
    3. Returns the patches as a 3D array (num_patches, p0, p1) ordered in column-major fashion.

    Args:
        data_array: The 2D input NumPy array (e.g., dtype uint8).
        patch_size: A tuple (p0, p1) specifying the patch dimensions.

    Returns:
        A 3D NumPy array (num_patches, p0, p1) containing all
        extracted patches in column-major order.
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
    # This groups by patch-column, then patch-row.
    transposed = reshaped.transpose(2, 0, 1, 3)
    
    # Reshape to flatten patch indices (num_patches_t, num_patches_n) into a single dimension, giving (total_patches, p0, p1).
    # This preserves the column-major order.
    patches = transposed.reshape(-1, p0, p1)

    return patches, num_patches_n

def load_tokenizer(path):
    print(f"Loading tokenizer from {path}...")
    
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)    
    return tokenizer

class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]): path to the dataset in the file system. If provided, data will be loaded from this path instead of downloaded.
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
        tokenizer_path: Optional[str] = None
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

        if isinstance(dataset_path, list):
            ds = concatenate_datasets([load_dataset(repo_name, split="train") for repo_name in dataset_path])
        else:
            ds = load_dataset(dataset_path, split="train")

        # NOTE: datasets are pre-shuffled
        self._data = split_dataset_by_node(ds, rank, world_size)
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.infinite = infinite
        self.tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path is not None else None

        # variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: List[int] = []

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample = np.array(sample['spike_counts'], dtype=np.uint8)

                if self.tokenizer is not None:
                    # Logic for tokenized data
                    # 1. Unpack both return values
                    patchified_sample, num_patches_n = get_patches_column_major(sample, self.tokenizer["patch_size"])
                    
                    # 2. Tokenize (results in a flat list of ints)
                    token_sequence = [self.tokenizer["index_map"].get(patch.tobytes(), 0) for patch in patchified_sample]

                    # 3. Vectorized insertion of special tokens
                    # Convert to numpy to use reshaping tricks
                    token_arr = np.array(token_sequence)
                    
                    # Reshape to (Time_Steps, Spatial_Patches_Per_Step)
                    # We use -1 for the time dimension to let numpy infer it automatically
                    token_arr = token_arr.reshape(-1, num_patches_n)

                    # Create the separator column (one per time step)
                    sep_token = self.vocab_size - 1
                    sep_col = np.full((token_arr.shape[0], 1), sep_token, dtype=token_arr.dtype)

                    # Stack horizontally: [SEP, Token1, Token2, ...]
                    token_arr = np.hstack((sep_col, token_arr))

                    # 4. Flatten back to a list and extend buffer
                    self._token_buffer.extend(token_arr.flatten().tolist())
                else:
                    # Logic for non-tokenized data
                    sample = np.concatenate((np.full((1, sample.shape[1]), self.vocab_size-1), sample), axis=0)
                    sample = sample.T.flatten().tolist()
                    self._token_buffer.extend(sample)

                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
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
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            # logger.info("Dataset is of type Dataset")
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))
        return iter(self._data)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


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
    tokenizer_path: Optional[str] = None
) -> DPAwareDataLoader:
    """
    Builds a data loader for distributed training.

    This function can create a data loader for a Hugging Face dataset or a
    dataset with synthetically generated data.

    Args:
        dataset_name (str): The name of the dataset. Use "synthetic" to generate data on the fly. Otherwise, use a name from _supported_datasets.
        batch_size (int): The batch size for the data loader.
        seq_len (int): The sequence length of the samples.
        vocab_size (int): The vocabulary size.
        world_size (int): The total number of processes in the distributed group.
        rank (int): The rank of the current process.
        dataset_path (Optional[str]): Path to a local dataset. Required for unsupported Hugging Face datasets.
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
            tokenizer_path=tokenizer_path
        )

    return DPAwareDataLoader(rank, dataset, batch_size=batch_size)
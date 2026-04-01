import argparse
import pickle
import numpy as np
from datasets import load_dataset

# Global variable so each HPC worker only loads the tokenizer into memory once
_worker_tokenizer = None

def process_example(example, tokenizer_path):
    global _worker_tokenizer
    # 1. Load the tokenizer locally within the worker
    if _worker_tokenizer is None:
        with open(tokenizer_path, 'rb') as f:
            _worker_tokenizer = pickle.load(f)
            
    tokenizer = _worker_tokenizer

    # 2. Extract raw data
    sample = np.array(example['spike_counts'], dtype=np.uint8)
    n, t = sample.shape
    
    p0, p1 = tokenizer["patch_size"]
    if p0 != 1:
        raise ValueError(f"This script assumes patch_size p0 == 1, but got {p0}")

    # 3. Pad the time dimension to be a multiple of p1 (15)
    pad_t = (p1 - (t % p1)) % p1
    if pad_t > 0:
        sample = np.pad(sample, ((0, 0), (0, pad_t)), mode='constant', constant_values=0)
    
    num_patches_t = sample.shape[1] // p1

    # 4. Reshape to isolate the (1, 15) patches
    reshaped = sample.reshape(n, num_patches_t, 1, p1)
    flat_patches = reshaped.reshape(-1, 1, p1)
    
    # 5. Tokenize
    token_sequence = [
        tokenizer["index_map"].get(patch.tobytes(), 0) 
        for patch in flat_patches
    ]
    
    # 6. Reshape back to 2D token array (Using uint16 for memory efficiency!)
    token_arr = np.array(token_sequence, dtype=np.uint16).reshape(n, num_patches_t)

    return {"tokenized_spikes": token_arr.tolist()}

def main():
    parser = argparse.ArgumentParser(description="Tokenize Neural Pile datasets offline (all splits).")
    parser.add_argument("--input_repo", type=str, default="eminorhan/neural-pile-rodent", help="Original repo location")
    parser.add_argument("--output_repo", type=str, default="eminorhan/neural-pile-primate-rodent-1x15", help="Tokenized repo location")
    parser.add_argument("--tokenizer_path", type=str, default="/lustre/blizzard/stf218/scratch/emin/gpt-neuro/tokenizers/tokenizer_rodent_1x15_32k.pkl")
    parser.add_argument("--num_proc", type=int, default=16)
    
    args = parser.parse_args()
    
    # Loading without the 'split' argument returns a DatasetDict containing all splits
    print(f"Loading all splits for dataset: {args.input_repo}...")
    dataset_dict = load_dataset(args.input_repo)
    
    # Get the column names from the first split so we know what raw data to remove
    first_split = list(dataset_dict.keys())[0]
    columns_to_remove = dataset_dict[first_split].column_names

    print(f"Found splits: {list(dataset_dict.keys())}")
    print("Applying tokenization across all splits...")
    
    # DatasetDict.map() automatically iterates through 'train', 'test', etc.
    processed_dataset_dict = dataset_dict.map(
        process_example,
        fn_kwargs={"tokenizer_path": args.tokenizer_path},
        num_proc=args.num_proc,
        remove_columns=columns_to_remove,
        desc="Tokenizing spike counts"
    )

    print(f"Pushing processed dataset to {args.output_repo}...")
    # This pushes the entire dictionary, preserving the 'train' and 'test' structure
    processed_dataset_dict.push_to_hub(args.output_repo, max_shard_size="1GB")
    print("Done!")

if __name__ == "__main__":
    main()
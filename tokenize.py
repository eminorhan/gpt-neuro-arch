import os
import numpy as np
from datasets import loade_dataset, Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders


# config (TODO: refactor these into argparse or something like that)
HF_REPO_NAME = "eminorhan/neural-pile-primate"  # hf repo to be tokenized (neural pile repos)
TARGET_VOCAB_SIZE = 128256  # target vocabulary size for the BPE model
BOS_TOKEN = 255  # bos token id
TOKENIZER_PATH = "spike-count-bpe-tokenizer.json"  # path to save the trained tokenizer file


# data iterator for tokenizer training
def get_training_corpus(dataset):
    """
    A generator that iterates through the dataset, applies the specific
    preprocessing you described, and yields samples as space-separated strings.
    The Hugging Face trainer will consume data from this generator.
    """
    for sample in dataset:
        # 1. Convert to numpy array.
        spike_counts = np.array(sample['spike_counts'])

        # 2. Prepend a row with the special token ID. Shape changes from (n, t) to (n+1, t).
        prefixed_sample = np.concatenate((np.full((1, spike_counts.shape[1]), BOS_TOKEN, dtype=np.uint8), spike_counts), axis=0)

        # 3. Transpose and flatten to get the final token sequence. Shape changes to (t, n+1) and then flattens to a 1D array.
        flattened_tokens = prefixed_sample.T.flatten()

        # The trainer expects text. We convert our integer tokens into a single string, with each token separated by a space.
        yield " ".join(map(str, flattened_tokens))

if __name__ == "__main__":

    spike_dataset = load_dataset(HF_REPO_NAME, split="train")
    print(f"Dataset ready with {len(spike_dataset)} samples.")

    # We will use a BPE model. <UNK> is for out-of-vocabulary tokens.
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    # The pre-tokenizer splits the input string into initial "words". Since we created space-separated strings of numbers, the Whitespace pre-tokenizer is the perfect choice.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define a few standard special tokens (TODO: is this necessary?)
    special_tokens = ["<UNK>", "<PAD>", "<CLS>", "<SEP>", "<MASK>"]

    # Set up BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        special_tokens=special_tokens,
        min_frequency=2,  # A merge will only be considered if the pair appears at least twice.
    )

    # train_from_iterator method is memory-efficient for large datasets
    tokenizer.train_from_iterator(get_training_corpus(spike_dataset), trainer=trainer, length=len(spike_dataset))  # Providing the length helps with progress tracking
    print("Training complete.")

    tokenizer.save(TOKENIZER_PATH)
    print(f"\nTokenizer saved successfully to '{TOKENIZER_PATH}'")

    # Take one sample from the dataset to test the tokenizer
    test_iterator = get_training_corpus(spike_dataset.select(range(1)))
    sample_text = next(test_iterator)
    original_token_count = len(sample_text.split())

    print(f"\nOriginal sample (first 80 tokens):")
    print(" ".join(sample_text.split()[:80]) + "...")

    # Encode the sample text
    encoding = tokenizer.encode(sample_text)
    encoded_token_count = len(encoding.ids)

    print(f"\nEncoded tokens (first 50):")
    print(encoding.tokens[:50])

    print(f"\nEncoded IDs (first 50):")
    print(encoding.ids[:50])

    print("\n--- Analysis ---")
    print(f"Final Vocabulary Size: {tokenizer.get_vocab_size()}")
    print(f"Original sequence length (number of integers): {original_token_count}")
    print(f"Encoded sequence length (after BPE merges):   {encoded_token_count}")

    if encoded_token_count < original_token_count:
        compression = (1 - encoded_token_count / original_token_count) * 100
        print(f"Tokenization resulted in a {compression:.2f}% sequence compression for this sample.")
    else:
        print("BPE did not find frequent pairs to merge in this specific sample.")
import os
import math
import numpy as np

from collections import defaultdict
from datasets import load_dataset, IterableDataset
from tqdm import tqdm

class NGramModel:
    """
    A class to build and evaluate an n-gram model for sequence data.

    This model works with sequences of integers, where each integer represents a
    token in a vocabulary. It calculates probabilities based on n-gram
    frequencies and can estimate the cross-entropy loss on a given dataset.
    This version is adapted to work with Hugging Face IterableDatasets.
    """
    def __init__(self, n, vocab_size=256):
        """
        Initializes the N-gram model.

        Args:
            n (int): The order of the n-gram model (e.g., 1 for unigram, 2 for bigram).
            vocab_size (int): The total number of unique tokens in the vocabulary. For uint8 data, this is 256.
        """
        if n < 1:
            raise ValueError("N must be at least 1.")
        self.n = n
        self.vocab_size = vocab_size
        
        # to store counts: self.counts[context][token] = count
        # context is a tuple of (n-1) tokens; for n=1, context is an empty tuple ()
        self.counts = defaultdict(lambda: defaultdict(int))
        
        # to store total counts for each context: self.context_counts[context] = total_count
        self.context_counts = defaultdict(int)

    def preprocess(self, sample):
        """Preprocess the data row"""
        sample = np.array(sample)
        sample = np.concatenate((np.full((1, sample.shape[1]), self.vocab_size-1), sample), axis=0)
        sample = sample.T.flatten().tolist()
        return sample

    def train(self, dataset: IterableDataset, sequence_key='tokens'):
        """
        Trains the n-gram model on an IterableDataset.

        This method iterates through all sequences in the dataset and counts the
        occurrences of each n-gram.

        Args:
            dataset (datasets.IterableDataset): A Hugging Face IterableDataset object.
            sequence_key (str): The name of the column containing the sequences.
        """
        print(f"Training {self.n}-gram model...")
        
        for item in tqdm(dataset):
            sequence = item[sequence_key]
            sequence = self.preprocess(sequence)

            # for unigram model, we just count individual tokens
            if self.n == 1:
                context = ()
                for token in sequence:
                    self.counts[context][token] += 1
                    self.context_counts[context] += 1
                continue

            # for n > 1, we slide a window across the sequence with window size n 
            # (the first n-1 tokens are the context and the last token is the one predicted)
            for i in range(len(sequence) - self.n + 1):
                context = tuple(sequence[i : i + self.n - 1])
                token = sequence[i + self.n - 1]
                self.counts[context][token] += 1
                self.context_counts[context] += 1

    def estimate_cross_entropy(self, dataset: IterableDataset, sequence_key='tokens', alpha=1.0):
        """
        Estimates the cross-entropy loss of the model on a given IterableDataset.

        Cross-entropy is calculated as the average negative log-likelihood of the
        sequences. We use Laplace (add-one) smoothing to handle unseen n-grams.

        Loss = - (1/T) * sum(log(P(token_i | context_i)))
        where T is the total number of tokens being predicted.

        Args:
            dataset (datasets.IterableDataset): The dataset to evaluate on.
            sequence_key (str): The name of the column containing the sequences.
            alpha (float): The smoothing parameter (1.0 for standard Laplace smoothing).

        Returns:
            float: The estimated cross-entropy loss in nats per token.
        """
        if not self.counts:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"Estimating cross-entropy for {self.n}-gram model...")
        
        total_log_prob = 0.0
        total_tokens_predicted = 0

        for item in tqdm(dataset):
            sequence = item[sequence_key]
            sequence = self.preprocess(sequence)
            # for unigram model
            if self.n == 1:
                context = ()
                total_context_count = self.context_counts.get(context, 0)
                denominator = total_context_count + alpha * self.vocab_size
                
                for token in sequence:
                    token_count = self.counts.get(context, {}).get(token, 0)
                    numerator = token_count + alpha
                    prob = numerator / denominator
                    
                    total_log_prob += math.log(prob)
                    total_tokens_predicted += 1
                continue

            # for n-gram model where n > 1
            for i in range(len(sequence) - self.n + 1):
                context = tuple(sequence[i : i + self.n - 1])
                token = sequence[i + self.n - 1]

                token_count = self.counts.get(context, {}).get(token, 0)
                total_context_count = self.context_counts.get(context, 0)

                numerator = token_count + alpha
                denominator = total_context_count + alpha * self.vocab_size
                
                prob = numerator / denominator
                
                total_log_prob += math.log(prob)
                total_tokens_predicted += 1

        if total_tokens_predicted == 0:
            return 0.0

        cross_entropy_loss = -total_log_prob / total_tokens_predicted
        return cross_entropy_loss


if __name__ == '__main__':

    # dataset info
    DATASET_PATH = "eminorhan/neural-pile-primate" 
    SEQUENCE_KEY = "spike_counts" # column name in the dataset that contains the data.

    # n-gram info
    N_VALUE = 1
    VOCAB_SIZE = 256
    
    # load the dataset
    print(f"Loading dataset '{DATASET_PATH}'...")
    ds_train = load_dataset(DATASET_PATH, split="train")
    ds_test = load_dataset(DATASET_PATH, split="test")
    print("Dataset loaded.")

    # initialize the n-gram model
    ngram_model = NGramModel(n=N_VALUE, vocab_size=VOCAB_SIZE)
    
    # train the model (NOTE: for large datasets, you might want to take a small subset for faster prototyping/debugging: e.g., ds_train.take(100))
    ngram_model.train(ds_train, sequence_key=SEQUENCE_KEY)
    
    # estimate the cross-entropy on test split
    loss = ngram_model.estimate_cross_entropy(ds_test, sequence_key=SEQUENCE_KEY)
    
    print("\n--- Results ---")
    print(f"Model: {N_VALUE}-gram")
    print(f"Test Cross-Entropy Loss: {loss:.4f} nats per token")
    print("----------------")
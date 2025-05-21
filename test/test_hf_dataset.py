
from datasets import load_dataset
import numpy as np

ds = load_dataset("eminorhan/willett", split="train", trust_remote_code=True)

print(f"Total size of the dataset (number of rows): {len(ds)}")

for sample in iter(ds):
    sample = sample['tx1']
    sample = np.array(sample).flatten()
    print(f"Sample: {sample}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample shape: {sample.shape}")
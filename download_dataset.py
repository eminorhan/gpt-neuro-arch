from datasets import load_dataset
import aiohttp
import numpy as np
import torch

ds = load_dataset("eminorhan/neural-pile-primate", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/neural-pile-primate", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/xiao", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/willett", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/churchland", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/neupane-entorhinal", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/kim", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/even-chen", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/wojcik", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/perich", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/makin", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/h2", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})

print(f"Number of data rows: {len(ds['train'])}")
raw_item = ds['train'][0]['spike_counts']
print(f"Shape via torch.tensor(): {torch.tensor(raw_item).shape}")
for d in ds["train"]:
    spike_counts = np.array(d['spike_counts'], dtype=np.uint8)
    print(f"Data row shape: {spike_counts.shape}")

print(f"Done!")
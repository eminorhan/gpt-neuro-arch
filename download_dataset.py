from datasets import load_dataset
import aiohttp
import numpy as np

# ds = load_dataset("eminorhan/neural-pile-primate", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
ds = load_dataset("eminorhan/neupane-ppc", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/willett", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/churchland", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/neupane-entorhinal", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
# ds = load_dataset("eminorhan/kim", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})

for i in range(10):
    spike_counts = np.array(ds['train'][i]['spike_counts'])
    print(f"Data row {i} shape: {spike_counts.shape}")

print(f"Done!")
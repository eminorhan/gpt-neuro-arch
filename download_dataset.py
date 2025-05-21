from datasets import load_dataset
import aiohttp

ds = load_dataset("eminorhan/neural-pile-primate", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
print(f"Done!")
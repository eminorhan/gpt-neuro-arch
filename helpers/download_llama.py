from huggingface_hub import hf_hub_download

hf_hub_download("meta-llama/Llama-3.1-8B", "params.json", subfolder="original", local_dir="outputs/checkpoint/")
hf_hub_download("meta-llama/Llama-3.1-8B", "consolidated.00.pth", subfolder="original", local_dir="outputs/checkpoint/")


from datasets import load_dataset

# ds = load_dataset("allenai/c4", name="realnewslike", split="train")
ds = load_dataset("eminorhan/neural-bench-rodent", download_mode='force_redownload')


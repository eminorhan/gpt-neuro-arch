from datasets import load_dataset

# ds = load_dataset("allenai/c4", name="realnewslike", split="train")
ds = load_dataset("eminorhan/neural-pile-primate", num_proc=64, download_mode='force_redownload')


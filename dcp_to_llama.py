import os
import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

class LogColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP to Llama.")
    parser.add_argument("--input_dir", type=Path, help="Input directory with DCP weights.")
    parser.add_argument("--output_dir", type=Path, help="Output directory for Llama weights.")
    parser.add_argument('--hf_repo_name',default="",type=str, help='the model will be pushed to this HF repo')    
    parser.add_argument('--push_to_hub', action='store_true', help='whether to push llama ckpt to hf hub (default: false)')
    args = parser.parse_args()

    # DCP_CKPT_DIR = "outputs/X/checkpoint/step-Y"  # input
    # LLAMA_CKPT_DIR = "outputs/tmp"  # output

    llama_path = os.path.join(args.output_dir, "checkpoint.pth")

    # convert dcp model to torch.save
    print(f"{LogColors.RED} DCP --> torch conversion {LogColors.GREEN} ({args.input_dir} --> {args.output_dir}) {LogColors.END}")
    dcp_to_torch_save(args.input_dir, llama_path)

    print(f"{LogColors.RED} Loading checkpoint with torch.load {LogColors.END}")
    x = torch.load(llama_path, map_location='cpu', weights_only=False)

    print(f"{LogColors.RED} Saving model state_dict only with torch.save {LogColors.END}")
    torch.save(x["model"], llama_path)

    if args.push_to_hub:
        print(f"{LogColors.RED} Pushing converted ckpt to hf hub {LogColors.END}")

        from huggingface_hub import HfApi

        api = HfApi()

        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.hf_repo_name,
            path_in_repo=args.input_dir.name,
            repo_type="model",
            token=True
        )
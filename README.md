### Large-scale distributed training of autoregressive generative models on *the Neural Pile*

This is a copy of the [`gpt-neuro`](https://github.com/eminorhan/gpt-neuro) repository on Arch, an HPE Cray EX254n system with 168 GH200 superchips. Currently the following models can be trained with this repository:

* `primate-8B-131k`: pretrained on primate data ([training script](train_primate_8B_131k.sh))

* `synthetic-8B-131k`: pretrained on synthetic data ([training script](train_synthetic_8B_131k.sh))

* `synthetic-primate-8B-131k`: pretrained on synthetic data -> finetuned on primate data ([training script](train_synthetic_primate_8B_131k.sh))

* `lang-primate-8B-131k`: pretrained on language -> finetuned on primate data ([training script](train_lang_primate_8B_131k.sh))

The training configurations for these models can be found in the [train_configs](train_configs) folder. These models are all trained with FSDP2 only. The larger GPU memory on GH200 makes it feasible to train the models with 131072 token context length without tensor parallelism (note that this is not possible on MI250X).

### Checkpoint conversions

To generate an initial checkpoint from the pretrained `llama-3.1-8B` model without copying the input and output layers (to take into account the different vocab size in our models):
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

To re-consolidate the `dcp` checkpoint into a single `.pth` checkpoint file and push it to HF Hub:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --hf_repo_name HF_REPO_NAME --push_to_hub
```

### Requirements

A successful reproduction requires the following steps.

Create a python virtual environment and activate it:
```bash
python -m venv myvenv
source myvenv/bin/activate
``` 

Install PyTorch with CUDA 12.6 support (Arch doesn't support later CUDA versions at the moment):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install the following packages:
```bash
pip install datasets torchdata tomli tensorboard blobfile tabulate ninja
```
and then you can clone this repo and run the training/evaluation/generation scripts.
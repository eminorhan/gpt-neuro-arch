### Large-scale distributed training of autoregressive generative models on *the Neural Pile*

This is a copy of the [`gpt-neuro`](https://github.com/eminorhan/gpt-neuro) repository on Arch, an HPE Cray EX254n system with 168 GH200 superchips. Currently, the following models can be trained with this repository:

* 250M, 870M, 2B, and 7B parameter models.
* Models with tokenized data (set `job_config.model.tokenizer_path` in the training config file) 
* Models with weighted loss (set `job_config.training.token_weights` in the training config file)
* Models with a fixed number of neurons *n* (set `job_config.training.n_fixed` in the training config file)

A variety of training config files can be found in the [train_configs](train_configs) folder. The code currently supports FSDP and TP for distributed training. 

### Training data

The training code in this repository trains the models on the *The Neural Pile* dataset. *The Neural Pile* is hosted on two public Hugging Face dataset repositories:
* [`eminorhan/neural-pile-primate`](https://huggingface.co/datasets/eminorhan/neural-pile-primate) hosts the primate data.
* [`eminorhan/neural-pile-rodent`](https://huggingface.co/datasets/eminorhan/neural-pile-rodent) hosts the rodent data.

You can download the data, *e.g.* using the `load_dataset` function in the Hugging Face `datasets` repository. You will need about 40 GB of free disk space in order to cache the primate data on disk and about 453 GB for the rodent data. The training code in this repository assumes that the dataset is already cached on local disk.

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

* Create a python virtual environment and activate it:
```bash
python -m venv myvenv
source myvenv/bin/activate
``` 

* Install PyTorch stable built with CUDA 12.9 (my torch version is `2.9.1+cu129`):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

* Install the following packages:
```bash
pip install datasets torchdata tomli tensorboard blobfile tabulate ninja wheel packaging
```

* Install FlashAttention-3 for the Hopper architecture as described [here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release), basically:
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

* Install the `aws-ofi-nccl` plugin, which will enable `nccl` to use `libfabric` (you need to change the paths below if you're not installing this on Arch):
```bash
wget https://github.com/aws/aws-ofi-nccl/releases/download/v1.17.2/aws-ofi-nccl-1.17.2.tar.gz
tar -xzvf aws-ofi-nccl-1.17.2.tar.gz
cd aws-ofi-nccl-1.17.2
CC=gcc CXX=g++ ac_cv_header_limits_h=yes ./configure --with-libfabric=/opt/cray/libfabric/2.2.0rc1 --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9 --enable-trace --prefix=/lustre/blizzard/stf218/scratch/emin/aws-ofi-nccl-1.17.2 --disable-tests
make
make install
```

* Then you can clone this repo and run the training and evaluation scripts here.
### Large-scale distributed training of autoregressive generative models on *the Neural Pile*

This is a copy of `gpt-neuro` on Arch, an HPE Cray EX254n system with 168 GH200 superchips. Currently re-training the following models on this system:

`primate-8B-131k`: pretrained on primate data

`lang-primate-8B-131k`: pretrained on language -> finetuned on primate data

These models are trained on 36 nodes (144 GPUs) with FSDP2 only. The larger GPU memory on GH200 makes it feasible to train the models with 131072 token context length without tensor parallelism (note that this is not possible on MI250X). The global batch size is 18.9M tokens per update (144 x 131072).

---

To generate an initial checkpoint from the pretrained `llama-3.1-8B` model without copying the input and output layers (to take into account the different vocab size in our models):
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

---

To re-consolidate the `dcp` checkpoint into a single `.pth` checkpoint file and push it to HF Hub:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --hf_repo_name HF_REPO_NAME --push_to_hub
```
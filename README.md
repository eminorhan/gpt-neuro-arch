### GPT-neuro: yet another foundation model for neural data

This is a copy of `gpt-neuro` on Arch, an HPE Cray EX254n system with 168 GH200 superchips.

---

`rodent-8B-131k`: pretrained on rodent data

`primate-8B-131k`: pretrained on primate data

`rodent-primate-8B-131k`: pretrained on rodent data -> finetuned on primate data

`text-primate-8B-131k`: pretrained on language -> finetuned on primate data

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
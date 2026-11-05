# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MLLM-Guard` given a file with generated results run through VLMEvalKit. Code is adapted from https://github.com/AI45Lab/MLLMGuard/tree/main/guardrank

## Installations

Download the requirements (on top of the original DMT requirements)

```
pip install pillow-avif-plugin # this is used to open .avif files with pillow when running inference
```

## 🗂️ Model Download

This evaluation depends on two trained evaluators. The first evaluator is a LoRA-finetuned LLaMa-2 model that evaluates Privacy, Bias, Toxicity, and Legality. The second evaluator is a finetuned Roberta-Large model that evaluates Hallucination.

Download the models from [Huggingface](https://huggingface.co/Carol0110/GuardRank/tree/main). The LoRA weights are in `guardrank-lora` and the Roberta-large weights are in `hallucination.pt`. 
Run the following command (from the `internvl_chat` subfolder) to download the models (recommended):

```bash
huggingface-cli download Carol0110/GuardRank --local-dir eval/mllmguard/GuardRank
```

or you can alternatively clone with git:
```
cd eval/mllmguard/
git clone https://huggingface.co/Carol0110/GuardRank
```

After downloading, the directory structure is:
```
eval/mllmguard
└── GuardRank
    └── hallucination.pt
    └── guardrank_lora
```

If your compute nodes do not have internet access, you also need the download the original models from Huggingface. 
Download them with the following:
```
# Roberta-Large
huggingface-cli download FacebookAI/roberta-large --local-dir eval/mllmguard/GuardRank/roberta-large

# LLaMA-2-7B
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir eval/mllmguard/GuardRank/Llama-2-7b-hf
```

> NOTE: To get access to Llama-2 7b, agree to the community license [here](https://huggingface.co/meta-llama/Llama-2-7b). 
> You'll need approval before using the model.

### Dataset Preparation

The dataset is on Huggingface, so you can download it with (recommended):
```
huggingface-cli download Carol0110/MLLMGuard --local-dir data/eval/mllmguard --repo-type dataset
```

Alternatively, you can again use git:

1. Make sure you have an HF Access Token for the `datasets/Carol0110/MLLMGuard` repo to have read-only permissions.

2. Download the dataset to your local machine (Note: when I tried to git clone directly to the cluster, the images with .JPG extensions wouldn't load for some reason...) and `rsync` it to the cluster at `data/mllmguard`

```
git lfs install
git clone https://<YOUR_HF_USERNAME>:<YOUR_HF_ACCESS_TOKEN>@huggingface.co/datasets/Carol0110/MLLMGuard
rm -rf MLLMGuard/.git # remove this to not cause issues when copying over the data to another repo
```

3. `rsync` the dataset to the cluster at `data/mllmguard/MLLMGuard`
```
rsync -aPz MLLMGuard/ <path to repo>/dmt/internval_chat/data/mllmguard/
```

**Note:**
Although VLMEvalKit runs inference on this dataset, it has a couple of errors:

- Skips 7 images for the hallucination subtask and 2 images for the privacy subtask
- Improperly runs the position-swapping subtask where it neglects running inference on the flipped images (i.e. half of the predictions are missing for this subtask)

This script additionally runs inference on Hallucination, Privacy and Position-Swapping before evalution.
Moreover, Hallucination and Noise-Injection evaluation requires access to the metadata, so that must be included in the dataset.

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following commands:

```bash
# Run inference on all splits (e.g. with 4 gpus, single-node setup supported only)
torchrun \
    --standalone \
    --nproc_per_node=4 \
    eval/mllmguard/run_inference.py \
    --model {MODEL_NAME} \
    --checkpoint {CHECKPOINT} \
    --dynamic \
    --seed {SEED} \
    --results_dir ${RESULTS_DIR} \
    --datasets all

# Compute metrics on all dimensions
# (can run on multiple gpus to perform distributed inference with LLaMA-2)
export HF_TOKEN="your_hf_token" 
torchrun \
    --standalone \
    --nproc_per_node=4 \
    eval/mllmguard/evaluate_mllmguard.py \
    --model {MODEL} \
    --checkpoint {CHECKPOINT} \
    --results_dir ${RESULTS_DIR} \
    --seed {SEED} \
    --datasets all
```


### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default         | Description                                                                                                       |
| ---------------- | ------ | --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--results_dir`   | `str`  | `'eval/mllmguard/internvl_1_5-26b_results'`            | Path where the manual inference results are saved.                                                                                     |
| `--datasets`     | `str`  | `'all'` | Dimension to evaluate (or all)                                                                     |
| `--model`      | `str` | no default, must be set         | Name of the model to evaluate 
| `--checkpoint`      | `str` | no default, must be set         | Model checkpoint to load for evaluation 
| `--mllmguard_data_dir`      | `str` | `'data/mllmguard/MLLMGuard/desensitize'`         | Where the MLLMGuard data is stored.                                                                    |
| `--save_dir`      | `str`  | identical to `--results_dir`             | Directory to save the metrics.                                                           |
| `--save_file_ext` | `str` | `'.csv'`         | The metric file extension (.xlsx or csv)                                                                       |
| `--encoder_model`         | `str` | `'eval/mllmguard/GuardRank/hallucination.pt'`         | Path to the hallucination model |
| `--decoder_model`         | `str` | `'eval/mllmguard/GuardRank/guardrank_lora'`         | Path to the LoRA-finetuned LlaMA model |
| `--verbose`         | `flag` | `False`         | Whether or not to log outputs to terminal |

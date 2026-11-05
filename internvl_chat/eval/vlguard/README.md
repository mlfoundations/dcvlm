# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `VLGuard`. Code is adapted from https://github.com/ys-zong/VLGuard/tree/main

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### VLGuard

The dataset must be downloaded locally (and not through HF `load_dataset` in order to match the images with the metadata). You first need to get access to the dataset through [HuggingFace](https://huggingface.co/datasets/ys-zong/VLGuard).

```
mkdir data/vlguard
```

Then, download the `test.zip` file locally and scp to the cluster (not sure if there is an easier way to do this!) and unzip it into `data/vlguard/test`

The annotations needed are in the `data/vlguard` folder.
The directory structure is:

```
data
└── gpt4_safe_safes.json
└── test.json
└── test/
    └── bad_ads/
    └── harm-p/
    └── hatefulMemes/
    └── HOD/
    └── privacyAlert/
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
export OPENAI_API_KEY="your_openai_api_key"
torchrun --nproc_per_node=8 eval/vlguard/evaluate_vlguard.py --checkpoint ${CHECKPOINT} --dynamic
```

Alternatively, you can run the following simplified command:

```shell
export OPENAI_API_KEY="your_openai_api_key"
GPUS=8 sh evaluate.sh ${CHECKPOINT} vlguard --dynamic
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default                | Description                                                                                                       |
| ---------------- | ------ | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`                   | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'safe_unsafes,safe_safes,unsafes'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`                | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`                    | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`                | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`                | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
| `--metaDir`         | `str` | `data/vlguard/test.json`                | Path to test images. |
| `--imageDir`         | `str` | `data/vlguard/test`                | Path to test annotations. |
| `--max_new_tokens`         | `int` | `256`                | Number of max tokens for generation. |
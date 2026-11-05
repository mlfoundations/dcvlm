# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MM-SafetyBench`.
For scoring, we use **GPT-4-0613** as the evaluation model. (Note that in the MM-SafetyBench paper, they query Azure's GPT-4, but doesn't specify the specific model)

Code is adapted from https://github.com/isXinLiu/MM-SafetyBench/tree/main

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MM-SafetyBench

```bash
mkdir -p data/eval/mmsafetybench
```

Download processed questions from [Github](`https://github.com/isXinLiu/MM-SafetyBench/tree/main/data`)
```bash
cd data/eval/mmsafetybench
git clone git@github.com:isXinLiu/MM-SafetyBench.git
mv MM-SafetyBench/data/processed_questions/ .
rm -rf MM-SafetyBench
```

Download images from [Google Drive](https://drive.google.com/file/d/1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_/view) and put images into `data/mmsafetybench/imgs`. You can do so programmatically with:

```bash
gdown https://drive.google.com/uc?id=1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_
unzip 'MM-SafetyBench(imgs).zip'
mv 'MM-SafetyBench(imgs)' imgs
rm 'MM-SafetyBench(imgs).zip'
```

After processing, the directory structure is:

```
data
└── mmsafetybench
    └── processed_questions/
        └── 01-Illegal_Acitivitiy.json
        └── 02-HateSpeech.json
    └── imgs/
        └── 01-Illegal_Acitivitiy/
        └── 02-HateSpeech/
        ...
```


## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command:

```bash
# run distributed inference (e.g., on 4 GPUs)
torchrun \
    --standalone \
    --nproc_per_node=4 \
    eval/mmsafetybench/evaluate_mmsafetybench.py \
    --model ${MODEL} \
    --checkpoint ${CHECKPOINT} \
    --seed $SEED \
    --dynamic \
    --results-dir ${EVAL_DIR}

# then compute metrics using GPT-4o-mini as a judge 
# NOTE: we might use a lighter one model
export OPENAI_API_KEY="your_key"
python eval/mmsafetybench/calculate_score.py \
    --model ${MODEL} \
    --checkpoint ${CHECKPOINT} \
    --seed $SEED \
    --results-dir ${EVAL_DIR} 
```


### Arguments
The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default                | Description                                                                                                       |
| ---------------- | ------ | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`                   | Path to the model checkpoint.                                                                                     |
| `--dynamic`      | `flag` | `False`                | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`                    | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`                | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`                | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
| `--splits`   | `str`  | `'SD,SD_TYPO,TYPO'`                   | Comma-separated list of splits.                                                                                   |
| `--scenarios`   | `str`  | `'01-Illegal_Activitiy,02-HateSpeech,03-Malware_Generation,04-Physical_Harm,05-EconomicHarm,06-Fraud,07-Sex,08-Political_Lobbying,09-Privacy_Violence,10-Legal_Opinion,11-Financial_Advice,12-Health_Consultation,13-Gov_Decision'`                   | Comma-separated list of scenarios.                                                                                     |
| `--questions-dir`         | `str` | `data/mmsafetybench/processed_questions`                | Path to questions. |
| `--images-dir`         | `str` | `data/mmsafetybench/imgs`                | Path to images. |
| `--out-dir`         | `str` | `eval/mmsafetybench/eval_results`                | Path to results directory. |
| `--questions-with-answers-dir`         | `str` | `eval/mmsafetybench/questions_with_answers`                | Path to questions with answers directory. |
| `--max_new_tokens`         | `int` | `256`                | Number of max tokens for generation. |
| `--max_new_gpt_tokens`         | `int` | `20`                | Number of max tokens for GPT to generate as a judge. |

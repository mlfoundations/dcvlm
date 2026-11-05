# DataComp-VLM

DataComp-VLM is a competition to curate the best training dataset to (pre)train Vision-Language Models. 
We release a standardized pool of up to 6T multimodal tokens, and participants can experiment with filtering, mixing, 
and formatting samples to create the best data. The resulting models are evaluated on a suite of upto 52 multimodal tasks.

## 📦 Releases

### Candidate pools

Here we provide links to the uncurated (raw) pools that each competition scale filters and mixes from. Pools are nested:
each smaller pool is a subset of the next.

| Scale | Pool size | 🤗 Hub |
| :---- | :-------- | :----- |
| `small`   | 187.5B tokens | [link](https://huggingface.co/datasets/mlfoundations/dcvlm_pool_small) |
| `medium`  | 750B tokens   | [link](https://huggingface.co/datasets/mlfoundations/dcvlm_pool_medium) |
| `large`   | 3T tokens     | [link](https://huggingface.co/datasets/mlfoundations/dcvlm_pool_large) |
| `x-large` | 6T tokens     | [link](https://huggingface.co/datasets/mlfoundations/dcvlm_pool_xlarge) |

### Pre-mixed, ready-to-train datasets
  
Here we provide read-to-train fully decontaminated, globally shuffled, and pre-mixed data mixtures — no curation or mixing needed, point any training stack at the shards and go!
Mixture is given as `captioning / instruction / text / multimodal-docs`, by training samples.

| Dataset | Scale | Mixture | Tokens | Samples | Shards | Size | 🤗 Hub |
| :------ | :---- | :------ | -----: | ------: | -----: | ---: | :----- |
| **DCVLM-Baseline** (6.25B) | `small` | 10 / 70 / 15 / 5 | 6.25B | 3,253,356 | 384 | 0.77 TB | [link](https://huggingface.co/datasets/mlfoundations/dcvlm-baseline-6_25b) |
| **DCVLM-Baseline** (200B) | `x-large` | 10 / 70 / 15 / 5 | 200B | 103,985,276 | 10,752 | 25.7 TB | [link](https://huggingface.co/datasets/mlfoundations/dcvlm-baseline-200b) |
| **DCVLM-Balanced** (200B) | `x-large` | 40 / 40 / 15 / 5 | 200B | 112,358,849 | 11,318 | 19.7 TB | [link](https://huggingface.co/datasets/mlfoundations/dcvlm-balanced-200b) |

### Models

| Model | Scale | Trained on | Vision init. | LLM init. | 🤗 Hub |
| :---- | :---- | :--------- | :----------- | :-------- | :----- |
| DCVLM-1B | `small`   | DCVLM-Baseline  | InternViT-300M | Qwen2.5-0.5B | [link](https://huggingface.co/mlfoundations/dcvlm-1b-model) |
| DCVLM-2B | `medium`  | DCVLM-Baseline  | InternViT-300M | Qwen2.5-1.5B | [link](https://huggingface.co/mlfoundations/dcvlm-2b-model) |
| DCVLM-4B | `large`   | DCVLM-Baseline  | InternViT-300M | Qwen2.5-3B   | [link](https://huggingface.co/mlfoundations/dcvlm-4b-model) |
| DCVLM-8B | `x-large` | DCVLM-Baseline  | InternViT-300M | Qwen2.5-7B   | [link](https://huggingface.co/mlfoundations/dcvlm-8b-model) |

NOTE: We are still releasing all our artefacts gradually at the links above. If a dataset / model has not yet been updated at their respective link, they are currently queued and will be updated as soon as possible. We appreciate your patience as we work through all our uploads! Our training, evaluation and data infrastructure code is in this repository.

## 🛠️ Getting Started

- **Installation:** see [INSTALLATION.md](INSTALLATION.md).
- **Data and models:** see [DATA.md](DATA.md) and [MODELS.md](MODELS.md).
- **Cluster configuration:** all jobs are submitted with SLURM and configured through `internvl_chat/.env` and `internvl_chat/scripts/slurm/system_setup.sh`; see [SLURM.md](SLURM.md).

## 🚀 Training

Training jobs are submitted with `internvl_chat/scripts/slurm/train/launcher.sh`. In *scale mode*, the `SCALE` variable selects the model, learning rate, batch size, warmup and total training tokens of a competition scale (see `internvl_chat/internvl/train/scale_configs.py`), and `META_JSON` selects your data mixture (relative to `META_DIR`, see [DATA.md](DATA.md#meta-json-file)):

```bash
cd internvl_chat
SCALE=small \
META_JSON=my_mixture.json \
NODES=8 \
JOB_NAME=my_run \
./scripts/slurm/train/launcher.sh
```

See [SLURM.md](SLURM.md) for all options, including a manual mode to set the training stage, model size and batch size yourself. Reference InternVL-2.5 training recipes are in `internvl_chat/scripts/train/`.

## 📊 Evaluation
The evaluation toolkit is a wrapper around [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main), and is located in `internvl_chat/dmt_evals`.
If you have installed training dependencies already, most libraries for evaluation should be there. Make sure to install the remaining with `pip install -r requirements/evaluation.txt`. 

**Brief overview.**
The entrypoint is `internvl_chat/dmt_evals/run.py`. Core arguments are:
- the `--data` argument, which is a (list of) benchmark(s) to evaluate on. See the [Supported Benchmarks](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY) on the official VLMEvalKit page.
- the `--model` argument, which is a (list of) model(s) to evaluate. Pass `--model DMT_InternVL2_5-{MODEL_SIZE}` to evaluate one of our models (e.g., `DMT_InternVL2_5-1B`).
- the `--load-dmt-pretrained` argument, which is a path to a checkpoint folder. 

**Interactive run.**
The following is an example of interactive execution (see [SLURM.md](SLURM.md) to submit eval jobs in batch mode), which evaluates a trained InternVL-2.5-4B model on the test set of AI2D from the checkpoint dumped at step 800 (i.e., assuming you have a checkpoint root at /my/ckpt/root and a run my_run, your checkpoint will be at /my/ckpt/root/my_run/checkpoint-800). Eval is distributed on 4 GPUs. You need your machine to be connected to the Internet to download a dataset.
```bash
cd internvl_chat

# import internvl would fail otherwise
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# needed by curr impl., may change in the future
export CKPT_DIR=/my/ckpt/root

# distributed inference w/ torchrun
torchrun --nproc_per_node=4 dmt_evals/run.py --model DMT_InternVL2_5-4B --data AI2D_TEST --load-dmt-pretrained /my/ckpt/root/my_run/checkpoint-800
```
If you want to evaluate a checkpoint at the end of training, which will be located at /my/ckpt/root/my_run, simply pass `--load-dmt-pretrained /my/ckpt/root/my_run`.

This will dump result files in `internvl_chat/outputs/DMT_InternVL2_5-4B/` named according to the dataset(s).

**Displaying results.**
`internvl_chat/dmt_evals/parse_all.py` collects the results of a checkpoint across our benchmark suite and prints them grouped by category (general, knowledge, OCR, vision-centric, multilingual, ...), flagging any benchmark whose results are missing:
```bash
cd internvl_chat
python dmt_evals/parse_all.py --model DMT_InternVL2_5-4B --ckpt my_run/checkpoint-800 --core --core_metric
```
- `--ckpt` is the checkpoint path relative to `CKPT_DIR`.
- `--validation`, `--core` or `--extended` restrict the results to one of the benchmark suites from the paper (13, 33 and 52 benchmarks); by default all supported benchmarks are included. Use `--data` to pick individual benchmarks instead.
- `--core_metric` reports a single score per benchmark, the average of every fully evaluated category, and, once all categories are complete, the overall average. Scores are also saved as CSV files in `./summaries/<ckpt>/` (or `--summary_dir`).
- `--use-judge` shows the scores obtained with LLM-as-a-judge answer extraction (see `SUBMIT_API` in [SLURM.md](SLURM.md)) instead of exact matching, and `--to-latex` prints the tables in LaTeX.
- Results are read from `./outputs` by default, where interactive runs save them. For evaluations submitted with SLURM, pass `--input_dir $EVAL_DIR --seed 1234` (the default seed of `multi_eval_launcher.py`).

**IMPORTANT NOTE.** The toolkit will download and save the datasets in the `LMUData` folder (`~/LMUData/` by default). If your compute nodes do not have internet access, you need to get the data beforehand (e.g., run *once* on a machine with internet access and transfer the folder, or use `internvl_chat/dmt_evals/download_and_log.py`).

## 🧪 Building the Pre-Mixed Datasets

The pre-mixed datasets above are built from a mixture's Meta-JSON in two steps: a *dump*, which runs our training data loader and writes every sample it produces to WebDataset shards, and a *reshuffle*, which mixes samples across shards so that every shard is a random sample of the whole dataset. To build your own, see [PREMIX_DATA.md](PREMIX_DATA.md).

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{farina2026datacomp,
  title={DataComp-VLM: Improved Open Datasets for Vision-Language Models},
  author={Farina, Matteo and Udandarao, Vishaal and Nguyen, Thao and Kuzucu, Selim and B{\"o}ther, Maximilian and Hochlehnert, Andreas and Ghosh, Adhiraj and Nezhurina, Marianna and Roth, Karsten and Struber, Joschka and others},
  journal={arXiv preprint arXiv:2606.28551},
  year={2026}
}
```

## Acknowledgements

This codebase is built with reference to the code of the following projects: [InternVL](https://github.com/OpenGVLab/InternVL), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [OpenCompass](https://github.com/open-compass/opencompass), [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [DINOv2](https://github.com/facebookresearch/dinov2), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), [Molmo](https://github.com/allenai/molmo), [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), and [NanoVLM](https://github.com/huggingface/nanoVLM). Thanks for their awesome work!

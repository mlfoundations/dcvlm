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

NOTE: We are still releasing all our artefacts gradually at the links above. If a dataset / model has not yet been updated at their respective link, they are currently queued and will be updated as soon as possible. We appreciate your patience as we work through all our uploads! We are also cleaning up our training, eval and data infra code, we will release those soon too!

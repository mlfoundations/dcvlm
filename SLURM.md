# Info for SLURM job submission

Here you can find info to submit both training and evaluation jobs with SLURM. 
We made an initial effort to make things as modular as possible, but feel free to contribute your improvements with a PR if you feel something is missing! 🤓


## 🛠️ Overview and initial setup
All files related to SLURM are located in `internvl_chat/scripts/slurm`. You must follow these steps before you submit any job:

**Step 1: Configure your `.env` file.** To avoid user- or machine-specific files subject to version control, all of your "local" variables for this project shall 
be in `internvl_chat/.env`. This includes:  
- **SLURM Accounting**: your account, partition, and QOS should be placed here. You can also configure a debugging setup (which is useful if, for example, your cluster has a "debug" QOS, or a partition with high priority access);
- **Hardware configuration**: You can configure the hardware details of your cluster here (how many gpus/cpus/memory per node), so the launcher will know how many resources to request for;
- **Local Directories**: where you wanna save your checkpoints, SLURM logs, evaluation files, wandb data, etc; 
- **API Keys for preprocessing and evaluation**: if your compute nodes have internet access, the evaluation toolkit can call API endpoints and use an LLM to extract answers from raw text. 
At the current stage, only the `OPENAI_API_KEY` is required.

A template with all the variables you need to set is located at `internvl_chat/.env.template`, so just run:
```bash
cp internvl_chat/.env.template internvl_chat/.env
``` 
and start editing with your changes in `internvl_chat/.env`. This file will be sourced from every submitted SLURM job, and ensures your configuration is consistent across different jobs.

**Step 2: Load your software stack with `system_setup.sh`.** Different clusters need different tweaks to run things smoothly. Similarly to above, just run 
```bash
cp internvl_chat/scripts/slurm/system_setup.sh.template internvl_chat/scripts/slurm/system_setup.sh
```
and start editing your changes in `internvl_chat/scripts/slurm/system_setup.sh` (ignored by version control). The template provides instructions about what modules you want to load here and examples for two different clusters. Specifically, here you need to:
- load your python environment; 
- make sure anything CUDA related is visible and properly setup;
- load a GCC version > 12 if that's not your default already.


Once again, the `internvl_chat/scripts/slurm/system_setup.sh` will be sourced from all SLURM submitted scripts and ensures your software stack is consistent across different jobs.

**Note.** `system_setup.sh` is sourced with one argument, `train`, `eval` or `text`, so you can keep separate environments for training, multimodal evaluation and text-only evaluation. For example:
```bash
# deepspeed may fail to build its extensions with an old gcc
module load gcc/12.2.0
if [ $1 == "train" ]; then
    source /path/to/envs/dcvlm-train/bin/activate
elif [ $1 == "eval" ]; then
    source /path/to/envs/dcvlm-eval/bin/activate
elif [ $1 == "text" ]; then
    source /path/to/envs/dcvlm-eval-text/bin/activate
else
    echo "Unknown environment type: $1. Expected 'train', 'eval' or 'text'."
    exit 1
fi
```

(Optional) If you're hitting deepspeed issues at startup, you can point `TORCH_EXTENSIONS_DIR` to a folder of precompiled torch extensions (put this line somewhere in `system_setup.sh`).

## 🚀 Training Jobs
Once everything is setup, you can submit training jobs with the entrypoint `scripts/slurm/train/launcher.sh`. There are two different "modes" for submitting jobs: the `scale` mode and the `manual` mode. 

### Scale Mode
This is the mode we'll be using for "official" runs, and spawns training according to a given compute scale. This mode is activated by setting the SCALE environment variable from cmd-line. Arguments:  
- **SCALE**: options are "debug", "small", "medium", "large", and "xlarge"; this determines which model to use, lr, batch size, warmup, and total training tokens;  
- **META_JSON**: the path, relative to your META_DIR (`.env`) of the JSON file collating data sources for training. For example, if you wanna use `$META_DIR/myannots.json`, set `META_JSON=myannots.json`; 
- **(Optional) NODES**: number of nodes to use for training. Default is 1; 
- **(Optional) MAX_SEQ_LEN**: maximum sequence length *and* maximum number of packed tokens when using data packing. Default is 16384;
- **(Optional) ESTIMATED_TOKENS**: expected number of valid tokens per packed sequence; If provided, it will determine the number of training steps to meet the total training tokens; 
- **(Optional) JOB_NAME**: name for the SLURM submission job, which will also coincide with the name of the run on W&B. A default name is automatically created, but we recommend setting a meaningful one; 
- **(Optional) DEBUG**: set to "0" to run with your default SLURM account, or set DEBUG=1 to use your debug configuration as per the `.env` file;
- **(Optional) TIME**: maximum wall clock time for the job. Default is 24hrs; ignored if DEBUG=1 (will use DEBUG_TIME from `.env`);

**Submission Example.**
```bash
 # *all* jobs must be submitted from internvl_chat
 cd internvl_chat

 # submit a job for the medium scale
 SCALE=medium \
 META_JSON=myannots.json \
 NODES=16 \
 JOB_NAME=my_wonderful_run \
 ./scripts/slurm/train/launcher.sh
```


### Manual Mode
If the SCALE variable is unset, you can still use the launcher by providing all explicit arguments:
- **TRAIN_STAGE**: set "1" for MLP only training, "2" for MLP+VisionEnc training, "3" for full model training;  
- **MODEL_SIZE**: "1B", "2B", or "4B" models. See [Models](./MODELS.md) to know what you need to download beforehand according to model size (or you can refer to the InternVL-2.5 paper);  
- **META_JSON**: the path, relative to `internvl_chat/data` of the JSON file which collates different data sources for training; 
- **(Optional) BATCH_SIZE**: *global* batch size. Default is 512; 
- **(Optional) PER_DEVICE_BATCH_SIZE**: Default is 1; Since we will likely be using automatic data packing, you don't need to set this > 1; 
- **(Optional) MAX_SEQ_LEN**: maximum sequence length *and* maximum number of packed tokens when using data packing. Default is 16384;
- **(Optional) JOB_NAME**: name for the SLURM submission job, which will also coincide with the name of the run on W&B. A default name is automatically created, but we recommend setting a meaningful one; 
- **(Optional) DEBUG**: set to "0" to run with your default SLURM account, or set DEBUG=1 to use your debug configuration as per the `.env` file;
- **(Optional) TIME**: maximum wall clock time for the job. Default is 24hrs; ignored if DEBUG=1 (will use DEBUG_TIME from `.env`);

**Submission Example.**
```bash
 # *all* jobs must be submitted from internvl_chat
 cd internvl_chat

 # submit the job (full model training, 1B model, 4 nodes, etc)
 TRAIN_STAGE=3 \
 MODEL_SIZE=1B \
 META_JSON=myannots.json \
 NODES=4 \
 JOB_NAME=my_wonderful_run \
 ./scripts/slurm/train/launcher.sh
```

**Behind the scenes**. The launcher will (i) make sure all required variables are set (STAGE, MODEL_SIZE, META_JSON); (ii) load your slurm, hw, local directories from `internvl_chat/.env`; (iii) submit a different `internvl_chat/scripts/slurm/*.sh` file according to model size and stage; (iv) at execution time, the software stack from your `internvl_chat/scripts/slurm/system_setup.sh` will be sourced and distributed training will be spinned.

### W&B logging
The `.env` file at `internvl_chat/.env` further specifies configs for W&B (`WANDB_TEAM`, `WANDB_MODE`, and the W&B directories). Each training job logs all its runs within a group identical to the JOB_NAME argument (in the previous example, all runs belong to the `my_wonderful_run` group). If your compute nodes have no internet access, keep `WANDB_MODE="offline"` and upload the runs afterwards with `wandb sync`.


## 📊 Evaluation Jobs

**IMPORTANT**: Make sure to follow the evaluation instructions at the bottom of [INSTALLATION.md](./INSTALLATION.md) to setup all the correct symlinks and paths. 

Similarly to training, evals have an entrypoint `scripts/slurm/eval/launcher.sh`. Evaluation works in single-node mode only. You'll need the following variables:  
- **MODEL**: keyword for the `--model` argument of the evaluation toolkit (see [Evaluating Runs](./README.md#-evaluation));
- **DATASETS**: whitespace-separated keywords for the `--data` argument of the evaluation toolkit (one inference job is submitted per dataset);
- **CKPT**: path to a pretrained ckpt folder, *relative to the CKPT_DIR variable of your `.env` file*. A valid ckpt folder has a config.json and a model.safetensors file.
- **SUBMIT_API**: flag to use API-based LLM-as-a-judge services (required for some benchmarks, official runs will need this set to "1"); 
- **(Optional) DEBUG**: identical to training; 
- **(Optional) TIME**: identical to training;

Here would be a job submission sample, following the example in [Evaluating Runs](./README.md#-evaluation):
```bash
# again, please launch this from the internvl_chat folder
cd internvl_chat

# eval a 4B model from ckpt at step 800 on AI2D
# NOTE: distribution on all gpus of a node is the default behaviour
MODEL=DMT_InternVL2_5-4B \
DATASETS=AI2D_TEST \
CKPT=my_run/checkpoint-800 \
./scripts/slurm/eval/launcher.sh
```

Note how, differently from the quickruns example, you don't need to export $CKPT_DIR, and the path of the ckpt is assumed to be relative to it. Evaluation files will be saved in your $EVAL_DIR (`.env`). 

**Recommended way of launching evals.** We have a convenient python launcher to submit all the supported multimodal evaluations of our benchmark. This is located at `internvl_chat/scripts/slurm/eval/multi_eval_launcher.py`, and just loops through supported benchmarks. You can launch all evals for a model with:
```bash
# always launch from internvl_chat
cd internvl_chat

# source the evaluation env
source scripts/slurm/system_setup.sh eval

# spawn jobs for all benchmarks 
python scripts/slurm/eval/multi_eval_launcher.py \
--model DMT_InternVL2_5-4B \
--ckpt my_run/checkpoint-800 
```

You can also set:
- a `--dataset` arg (accepts whitespace concatenation thorugh nargs="+") to submit only selected datasets; 
- the `--debug_mode` flag (equivalent to setting the DEBUG=1 environment variable); 
- the `--submit_api` flag (equivalent to setting the SUBMIT_API=1 environment variable).

For example:
```bash
# always launch from internvl_chat
cd internvl_chat

# source the evaluation env
source scripts/slurm/system_setup.sh eval

# spawn jobs only for mmmu and ai2d, ensuring llm-as-a-judge is used
python scripts/slurm/eval/multi_eval_launcher.py \
--model DMT_InternVL2_5-4B \
--ckpt my_run/checkpoint-800 \
--dataset MMMU_DEV_VAL AI2D_TEST \
--debug_mode \
--submit_api
```

**Note.** When `SUBMIT_API=1`, the LLM-as-a-judge scoring of all datasets passed in one launcher call runs as a single job on `INTERNET_PARTITION`. `multi_eval_launcher.py` passes `--api_every` datasets per call (default 10), which keeps the number of jobs on that partition low.


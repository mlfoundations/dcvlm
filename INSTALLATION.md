## 🛠️ Installation

- Clone this repository:

  ```bash
  git clone https://github.com/mlfoundations/dcvlm.git
  cd dcvlm
  ```

- Create a virtual environment and activate it:

  ```bash
  conda create -n dcvlm python=3.11 -y
  conda activate dcvlm
  ```

- Install dependencies:

  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements/training.txt
  pip install -r requirements/data.txt
  pip install webdataset wheel  # flash-attn needs wheel to build
  ```

### Additional Instructions

- Install `flash-attn`:

  ```bash
  pip install flash-attn --no-build-isolation
  ```

  Alternatively you can compile from source:

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v2.3.6
  python setup.py install
  ```

- Install `apex` (optional):

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

  If you encounter `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`, it is because apex's CUDA extensions are not being installed successfully. You can try uninstalling apex and the code will default to the PyTorch version of RMSNorm. Alternatively, if you prefer using apex, try adding a few lines to `setup.py` and then recompiling.

- Install and sign in to `wandb` for logging

  ```bash
  pip install wandb
  wandb login
  ```

### Troubleshooting

- Ensure that your pytorch and cuda versions are exactly the same. If they are not, `flash_attn` and other packages might error out. You can check this with `import torch; torch.version.cuda` in python and `nvcc --version` on commandline. We have tested this across multiple clusters with cuda:12.1.

- Ensure that your GCC toolkit is >=9, otherwise building `FusedAdam` with deepspeed might not work well. You can check this with `gcc --version`.

- Ensure that you use the correct version of deepspeed specified in the requirements (`0.15.4`), else you could potentially run into some gnarly issues with gradient accumulation, see [here](https://github.com/deepspeedai/DeepSpeed/issues/6793).

- Sometimes the deepspeed `FusedAdam` build might break because of certain mismatches between cuda and pytorch, and/or due to incorrect linking of library files, in those cases, setting these environment variables correctly might be helpful: `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`, `CPATH`, `CFLAGS`, `LDFLAGS`.

- Installation errors, particularly with respect to mismatched cuda versions, can likely be solved by cleanly creating the environment after a `pip cache purge`.

- If `fused_adam` fails to build with `cannot find -lcudart: No such file or directory`, ensure the CUDA runtime library is visible to the linker. In Conda-based CUDA setups, the actual `libcudart.so` may reside under `$CONDA_PREFIX/targets/x86_64-linux/lib` while being symlinked from `$CONDA_PREFIX/lib`. At other times, the symlink of `libcudart.so` might be incorrect. Make sure both paths are included in `LD_LIBRARY_PATH`, then clear `~/.cache/torch_extensions` and retry.

- If you get errors like `loaded_dict = pickle.load(handle); EOFError: Ran out of input` upon importing deepspeed, clear your triton cache (typically `~/.triton`) and try again.

### Evaluation
For evaluation, you can re-use the same Python environment used for training, since most libraries are installed already. Make sure you install the remaining with: `pip install -r requirements/evaluation.txt`.

Some benchmarks (MLLMGuard, VLGuard, MM-SafetyBench, Mantis-Eval, MMVP) are evaluated with the scripts in `internvl_chat/eval/` and expect their data under `internvl_chat/data/eval/`. Download each of them as described in its README in `internvl_chat/eval/<benchmark>/`, then link them in:

```bash
# run these from the repo root
source internvl_chat/.env

mkdir -p internvl_chat/data/eval
EVAL_ROOT=/path/to/your/eval_datasets

# MLLMGuard (+ evaluator model checkpoints)
ln -s $EVAL_ROOT/MLLMGuard internvl_chat/data/eval/mllmguard
huggingface-cli download Carol0110/GuardRank --local-dir internvl_chat/eval/mllmguard/GuardRank
ln -s $OFFLINE_MODEL_DIR/Llama-2-7b-hf internvl_chat/eval/mllmguard/GuardRank/Llama-2-7b-hf
ln -s $OFFLINE_MODEL_DIR/roberta-large internvl_chat/eval/mllmguard/GuardRank/roberta-large

# VLGuard
ln -s $EVAL_ROOT/VLGuard internvl_chat/data/eval/vlguard

# MM-SafetyBench
ln -s $EVAL_ROOT/mmsafetybench internvl_chat/data/eval/mmsafetybench

# Mantis-Eval
ln -s $EVAL_ROOT/TIGER-Lab/Mantis-Eval internvl_chat/data/eval/Mantis-Eval

# MMVP
ln -s $EVAL_ROOT/MMVP internvl_chat/data/eval/MMVP
```

Additionally, set these in your `internvl_chat/.env` file:
1. `LMUData`, the data directory used by VLMEvalKit (benchmarks are downloaded there on first use).
2. `COMPASS_DATA_CACHE`, the data directory used by OpenCompass for the text-only evaluations.

**Note.** For some datasets (e.g., Mantis-Eval) HF Datasets rewrites into Arrow format during the first execution, in the default `~/.cache` location. If your `$HOME` has little disk quota, set `$HF_HOME` to a location with more space.

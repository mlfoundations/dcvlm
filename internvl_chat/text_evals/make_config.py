"""
Inputs (through argparse):
- dataset name; 
- model name; 
- checkpoint path;

Output:
- A .py file to serve as the config for `opencompass`.
"""

import sys
import os
import os.path as osp
from dotenv import load_dotenv

# recurse up to the "internvl_chat" dir
parent_dir = osp.dirname(osp.abspath(__file__))
while osp.basename(parent_dir) != "internvl_chat":
    parent_dir = osp.dirname(parent_dir)

# add the internvl_chat dir to PATH
# (this is likely unnecessary)
ENV_ROOT = parent_dir
if ENV_ROOT not in sys.path:
    sys.path.append(ENV_ROOT)

TEXTEVAL_DIR = osp.join(ENV_ROOT, "text_evals")
TEXT_DATASET_ROOT = osp.join(TEXTEVAL_DIR, "dataset")

def read_file(path="./template.py"):
    if not osp.isfile(path):
        raise ValueError(
            f"Template file {path} does not exist."
        )
    with open(path, "r") as f:
        content = f.read()
    return content


def available_datasets(root=TEXT_DATASET_ROOT):
    py_files = os.listdir(root)
    return sorted([
        f.replace(".py", "") for f in py_files if f.endswith(".py")
        and (f != "__init__.py")
    ])


def main(
    model: str, 
    dataset: str, 
    ckpt_base: str, 
    ckpt_root: str, 
    max_tokens: int
):
    # load internvl_chat/.env file
    load_dotenv(osp.join(ENV_ROOT, '.env'))

    # load absolute model path
    # preferential loading through --ckpt-dir, will search for CKPT_DIR otherwise
    if not ckpt_root:
        model_dir = os.getenv("CKPT_DIR")
    else:
        model_dir = ckpt_root
    
    if model_dir is None or not osp.isdir(model_dir):
        raise ValueError(
            "--ckpt-dir is not given and "
            "CKPT_DIR environment variable is not set. "
            "Please set it to the directory containing the model checkpoints."
        )
    model_path = osp.join(model_dir, ckpt_base)

    # reads template, and formats it according to args
    template = read_file(path=osp.join(TEXTEVAL_DIR, "template.py"))

    # init datasets according to args
    avail_datasets = available_datasets()
    if dataset not in avail_datasets:
        raise ValueError(
            f"Dataset {dataset} is not available. "
            f"Available datasets: {avail_datasets}"
        )
    
    # swap in dataset details
    dataset_import = read_file(path=osp.join(TEXT_DATASET_ROOT, f"{dataset}.py"))
    dataset_names = dataset_import.split("import")[-1].strip()
    template = template.replace("{{DATASET_IMPORT}}", dataset_import)
    template = template.replace("{{DATASET_NAMES}}", dataset_names)

    # swap in model details
    template = template.replace("{{MODEL_PATH}}", f'"{model_path}"')

    # swap in model abbreviation
    abbr = f"{model}-{ckpt_base}"
    template = template.replace("{{ABBR}}", f'"{abbr}"')

    # swap in generation config 
    # TODO: potentially extend, but likely unneeded
    template = template.replace("max_tokens = 256", f"max_tokens = {max_tokens}")

    # save the config to the outfile
    config_dir = osp.join(TEXTEVAL_DIR, "configs", model, ckpt_base)
    os.makedirs(config_dir, exist_ok=True)
    config_path = osp.join(config_dir, f"{dataset}_config.py")
    with open(config_path, "w") as f:
        f.write(template)
    print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model Name.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset names")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--ckpt-dir", type=str, required=False, default=None, help="Directory containing model checkpoints")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens for generation")
    args = parser.parse_args()
    main(
        model=args.model,
        dataset=args.dataset,
        ckpt_base=args.ckpt,
        ckpt_root=args.ckpt_dir,
        max_tokens=args.max_tokens
    )
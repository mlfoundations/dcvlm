import os
import os.path as osp
import subprocess
import sys
from argparse import ArgumentParser
from dotenv import load_dotenv

# make sure we launch from the `internvl_chat` root directory
eval_toolkit_dir = osp.join(os.getcwd(), "dmt_evals")
error_msg = (
    f"Expected to find eval toolkit at {eval_toolkit_dir}, but it does not exist. "
    f"Are you running this script from the `internvl_chat` root directory?"
)
assert osp.isdir(eval_toolkit_dir), error_msg
sys.path.append(os.getcwd())
sys.path.append(eval_toolkit_dir)

from vlmeval import supported_VLM
from vlmeval.dataset import (
    FLAT_DMT_DATASETS, 
    FLAT_DMT_VALIDATION_DATASETS, 
    FLAT_DMT_CORE_DATASETS, 
    FLAT_DMT_EXTENDED_DATASETS
)
from vlmeval.dataset import (
    DMT_SUPPORTED_DATASETS,
    DMT_VALIDATION_DATASETS,
    DMT_CORE_DATASETS,
    DMT_EXTENDED_DATASETS
)
from vlmeval.utils.dmt_utils import get_score


def check_args(args, avail_datasets):
    if args.model is None or args.ckpt is None:
        sys.exit("\nERROR: You must set both MODEL and CKPT environment variables.")
    if args.model not in supported_VLM:
        sys.exit(f"\nERROR: model {args.model} is not supported.")
    # NOTE: exclude path check for "random" checkpoints
    is_random = (
        "1B-Init" in args.ckpt 
        or "2B-Init" in args.ckpt 
        or "4B-Init" in args.ckpt
    )
    ckpt_abspath = osp.join(os.getenv("CKPT_DIR"), args.ckpt)
    if not osp.isdir(ckpt_abspath) and not is_random:
        sys.exit(f"\nERROR: {ckpt_abspath} doesn't appear to be a valid ckpt directory.")
    if args.dataset:
        for dataset in args.dataset:
            if dataset not in avail_datasets:
                sys.exit(f"\nERROR: dataset {dataset} is not supported.")


def main(args):
    # load from .env file
    load_dotenv()
    
    # restrict to the subset of evals according to cmd-line args
    if args.validation:
        FLAT_DATASETS = FLAT_DMT_VALIDATION_DATASETS
        SUPPORTED_DATASETS = DMT_VALIDATION_DATASETS
    elif args.core:
        FLAT_DATASETS = FLAT_DMT_CORE_DATASETS
        SUPPORTED_DATASETS = DMT_CORE_DATASETS
    elif args.extended:
        FLAT_DATASETS = FLAT_DMT_EXTENDED_DATASETS
        SUPPORTED_DATASETS = DMT_EXTENDED_DATASETS
    else:
        FLAT_DATASETS = FLAT_DMT_DATASETS
        SUPPORTED_DATASETS = DMT_SUPPORTED_DATASETS

    # override full list of benchmarks if passed from command line
    datasets = args.dataset if args.dataset else FLAT_DATASETS
    
    # system checks
    check_args(args, avail_datasets=FLAT_DATASETS)
    print("Checks completed, launching!")
    
    # launch all combos
    to_submit = []
    for i, dataset in enumerate(datasets):

        # do not submit if belonging to --exclude_category
        should_skip = False
        if args.exclude_category is not None:
            for exclude_cat in args.exclude_category:
                exclude_datasets = SUPPORTED_DATASETS.get(exclude_cat, [])
                if dataset in exclude_datasets:
                    print(f"Skipping {dataset} as it belongs to --exclude_category {exclude_cat}")
                    should_skip = True
                    break

        # but override if belonging to --include_category
        if args.include_category is not None:
            should_skip = True
            for include_cat in args.include_category:
                include_datasets = SUPPORTED_DATASETS.get(include_cat, [])
                if dataset in include_datasets:
                    print(f"Forcing inclusion of {dataset} as it belongs to --include_category {include_cat}")
                    should_skip = False
                    break

        if should_skip:
            continue

        # lastly, skip if results for this combo exist already
        # (this is useful when launching, e.g., "extended" after "core")
        results_exist = False
        try:
            eval_root = os.getenv("EVAL_DIR", None)
            score = get_score(
                model=args.model,
                dataset=dataset,
                ckpt=args.ckpt,
                eval_root=f"{eval_root}/seed_{args.seed}",
                suppress_warnings=True,
                core_metric_only=True,
                use_judge=args.api_only
            )
            results_exist = True
        except Exception:
            pass

        if results_exist and not args.override:
            print(f"Skipping {dataset} as results already exist. ({score=})")
            continue

        # if we get here, we can submit this dataset
        # NOTE: 11/30, due to Leonardo's updated policies
        # we cannot submit more than 10 jobs overall to the lrd_all_serial
        # partitition for API Evals, so we "batch" the datasets
        to_submit.append(dataset)
        if len(to_submit) == args.api_every:
            # create a copy of the necessary env-vars
            env = os.environ.copy()
            env.update({
                "DEBUG": str(int(args.debug_mode)),
                "MODEL": args.model,
                "DATASETS": " ".join(to_submit),
                "CKPT": args.ckpt,
                "SEED": str(args.seed),
                "SUBMIT_API": str(int(args.submit_api)),
                "API_ONLY": str(int(args.api_only)),
            })

            # submit the current batch of datasets
            print(f"[{i+1}/{len(datasets)}] Launching eval for {to_submit}...")
            if not args.dry_run:
                subprocess.run(["bash", "./scripts/slurm/eval/launcher.sh"], env=env)
            
            # reset the aggregator list 
            to_submit = []
    
    # submit the remaining datasets
    if to_submit:
        env = os.environ.copy()
        env.update({
            "DEBUG": str(int(args.debug_mode)),
            "MODEL": args.model,
            "DATASETS": " ".join(to_submit),
            "CKPT": args.ckpt,
            "SEED": str(args.seed),
            "SUBMIT_API": str(int(args.submit_api)),
            "API_ONLY": str(int(args.api_only)),
        })

        # submit the current batch of datasets
        print(f"[{i+1}/{len(datasets)}] Launching eval for {to_submit}...")
        if not args.dry_run:
            subprocess.run(["bash", "./scripts/slurm/eval/launcher.sh"], env=env)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--dataset", type=str, nargs="+", default=None)
    parser.add_argument("--include_category", type=str, nargs="+", default=None)
    parser.add_argument("--exclude_category", type=str, nargs="+", default=None)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--submit_api", action="store_true")
    parser.add_argument("--api_every", type=int, default=10)
    parser.add_argument("--api_only", action="store_true")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--override", action="store_true")


    args = parser.parse_args()
    main(args)
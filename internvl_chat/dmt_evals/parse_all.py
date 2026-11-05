import os
import sys
import json
import pandas as pd
import os.path as osp
from pathlib import Path
from rich.console import Console
from argparse import ArgumentParser
from tabulate import tabulate

# in-code hack to avoid the "export PYTHONPATH" statement
# adds the internvl_chat folder to $PATH
dirname = osp.dirname(osp.dirname(osp.abspath(__file__)))
if dirname not in sys.path:
    sys.path.append(dirname)
from vlmeval.config import supported_VLM
from vlmeval.dataset import (
    DMT_SUPPORTED_DATASETS, 
    DMT_CORE_DATASETS, 
    DMT_VALIDATION_DATASETS, 
    DMT_EXTENDED_DATASETS,
    FLAT_DMT_DATASETS,
    FLAT_DMT_CORE_DATASETS,
    FLAT_DMT_VALIDATION_DATASETS,
    FLAT_DMT_EXTENDED_DATASETS,
)
from vlmeval.utils.dmt_utils import get_score

NORM_CONSTANTS = {
    "MME": 2800,
    "OCRBench": 1000, 
    "COCO_VAL": 1000, 
}

def float_convert(df):
    for col in df.columns:
        try:
            df[col] = df[col].float()
        except:
            pass
    return df


def json_to_csv(path):
    with open(path, "r") as file_ptr:
        ddict = json.load(file_ptr)
    for k, v in ddict.items():
        if not isinstance(v, list):
            ddict[k] = [v]
    df = pd.DataFrame.from_dict(ddict)
    return df


def get_ckpt_from_path(path: str):
    splits = path.split("<ckpt>")
    return splits[1]

def avg(lst):
    return sum(lst) / len(lst)

def get_micro_avg(successful: dict, return_flat_breakdown: bool = True):
    scores = []
    datasets = []
    for dataset, score_dict in successful.items():
        curr_score = list(score_dict.values())[0]
        scaling = NORM_CONSTANTS.get(dataset, 100)
        scores.append(100*curr_score/scaling)
        datasets.append(dataset)
    if return_flat_breakdown:
        return datasets, scores, sum(scores) / len(scores)
    return sum(scores) / len(scores)

def tabulate_category(category, scores, precision=4, latex=False):
    headers = []
    cat_scores = []
    for d in SUPPORTED_DATASETS[category]:
        curr_score = list(scores[d].values())[0]
        scaling = NORM_CONSTANTS.get(d, 100) 
        cat_scores.append(100*curr_score/scaling)
        headers.append(d)

    cat_scores.append(avg(cat_scores))
    headers.append("Average")
    tablefmt = "latex" if latex else "grid"
    print(tabulate([cat_scores], headers=headers, tablefmt=tablefmt, floatfmt=f".1f"))
    return cat_scores[-1]
    


def main(args):

    model = args.model 
    ckpt = args.ckpt
    input_dir = args.input_dir
    if args.seed:
        input_dir = osp.join(input_dir, f"seed_{args.seed}")
    console = Console()

    assert model in supported_VLM, f"Model {model} not supported. Supported models: {supported_VLM}"

    # restrict to subset of evals according to args
    global SUPPORTED_DATASETS
    global FLAT_DATASETS
    if args.validation:
        SUPPORTED_DATASETS = DMT_VALIDATION_DATASETS
        FLAT_DATASETS = FLAT_DMT_VALIDATION_DATASETS
    elif args.core:
        SUPPORTED_DATASETS = DMT_CORE_DATASETS
        FLAT_DATASETS = FLAT_DMT_CORE_DATASETS
    elif args.extended:
        SUPPORTED_DATASETS = DMT_EXTENDED_DATASETS
        FLAT_DATASETS = FLAT_DMT_EXTENDED_DATASETS
    else:
        SUPPORTED_DATASETS = DMT_SUPPORTED_DATASETS
        FLAT_DATASETS = FLAT_DMT_DATASETS


    datasets = list([d for d in args.data]) if args.data else FLAT_DATASETS

    # loop through model/dataset/ckpt triplets combos
    missing = {}
    successful = {}
    for dataset in datasets:
        try:
            score = get_score(
                model,
                dataset,
                ckpt,
                eval_root=input_dir,
                core_metric_only=args.core_metric,
                suppress_warnings=True,
                use_judge=args.use_judge
            )

            for k, v in score.items():
                score[k] = round(v, args.precision)

            successful[dataset] = score

        except Exception as e:
            missing[dataset] = e

    # log based on category membership
    complete_categories, missing_categories = [], []
    for cat, cat_datasets in SUPPORTED_DATASETS.items():
        found = 0
        statements = []
        for d in cat_datasets:
            if d in successful:
                statements.append(f"Dataset={d}, Score={successful[d]}")
                found += 1
            elif d in missing:
                statements.append(f"[red]Dataset={d} failed with Exception={missing[d]}")
        
        if statements:
            console.print(f"\n[bold]{'-'*50} {cat} {'-'*50}")
            for stmt in statements:
                console.print(stmt)
        
        if found == len(cat_datasets):
            console.print(f"[green]All datasets in category ``{cat}`` completed successfully.")
            complete_categories.append(cat)
        else:
            missing_categories.append(cat)

    if args.core_metric:
        datasets, scores, _ = get_micro_avg(successful)
        os.makedirs(osp.join(args.summary_dir, ckpt), exist_ok=True)
        df = pd.DataFrame.from_dict({d: [s] for d, s in zip(datasets, scores)}).round(1)
        suffix = "judge" if args.use_judge else "exact"
        df.to_csv(osp.join(args.summary_dir, ckpt, f"per_dataset_{suffix}.csv"), index=False)

    # log completion summary if trying to parse everything
    if len(datasets) == len(FLAT_DATASETS) and args.core_metric:
        if complete_categories:
            console.print(f"\n[bold]{'-'*50} COMPLETE CATEGORIES {'-'*50}")
            cat_avgs = []
            cat_headers = []
            for cat in complete_categories:
                console.print(f"[green]{cat}")
                # log category average
                if args.core_metric:
                    cat_avg = tabulate_category(
                        category=cat, 
                        scores=successful, 
                        precision=args.precision,
                        latex=args.to_latex,
                    )
                    cat_avgs.append(cat_avg)
                    cat_headers.append(cat)

        if missing_categories:
            console.print(f"\n[bold]{'-'*50} MISSING CATEGORIES {'-'*50}")
            for cat in missing_categories:
                console.print(f"[red]{cat}")
        else:
            console.print(f"\n[bold][green]All categories completed successfully.")
            cat_avgs.append(avg(cat_avgs))
            cat_headers.append("Avg. of Avgs")
            tablefmt = "latex" if args.to_latex else "grid"
            # micro-average taken over all datasets
            cat_headers.append("Micro Avg.")
            datasets, scores, micro_avg = get_micro_avg(successful)
            cat_avgs.append(micro_avg)
            # display per-category breakdown
            print(tabulate([cat_avgs], headers=cat_headers, tablefmt=tablefmt, floatfmt=f".1f"))
            # dump summary on disk
            df = pd.DataFrame.from_dict({cat: [avg] for cat, avg in zip(cat_headers, cat_avgs)}).round(1)
            suffix = "judge" if args.use_judge else "exact"
            df.to_csv(osp.join(args.summary_dir, ckpt, f"per_category_{suffix}.csv"), index=False, sep="\t")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary_dir", type=str, default="./summaries")
    parser.add_argument("--data", type=str, nargs="+", help="Names of Datasets")
    parser.add_argument("--model", type=str, help="Name of Model")
    parser.add_argument("--ckpt", type=str, help="Checkpoints to evaluate")
    parser.add_argument("--core_metric", action="store_true", default=False)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--validation", action="store_true", default=False, help="Restrict to validation datasets")
    parser.add_argument("--core", action="store_true", default=False, help="Restrict to core datasets")
    parser.add_argument("--extended", action="store_true", default=False, help="Restrict to extended datasets")
    parser.add_argument(
        "--use-judge", action="store_true", default=False, 
        help="Display LLM-as-a-Judge results. Exact Matching is displayed by default."
    )
    parser.add_argument("--to-latex", action="store_true", default=False)
    
    args = parser.parse_args()
    main(args)

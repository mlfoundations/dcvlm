"Organizes the results into dataset-specific .tsv files, where rows are sorted from best to worst model."

import os
import sys
import json
import pandas as pd
import os.path as osp
from glob import glob
from argparse import ArgumentParser

# in-code hack to avoid the "export PYTHONPATH" statement 
# adds the internvl_chat folder to $PATH
dirname = osp.dirname(osp.dirname(osp.abspath(__file__)))
if dirname not in sys.path: sys.path.append(dirname)
from vlmeval.config import supported_VLM
from vlmeval.dataset import SUPPORTED_DATASETS


SORTING_KEYS = {
    'MMMU_DEV_VAL': ['split', 'Overall'],
    'MMBench_DEV_EN': ['split', 'Overall'],
    'MMBench_DEV_CN': ['split', 'Overall'],
    'AI2D_TEST': ['Overall'],
    'HallusionBench': ['split', 'aAcc'],
    'TextVQA_VAL': ['Overall'],
    'ScienceQA_VAL': ['split', 'Overall'],
    'OCRBench': ['Final Score'],
    'MathVision_MINI': ['Subject', 'acc'],
    'MathVista_MINI': ['Task&Skill', 'acc'],
    'SEEDBench_IMG': ['Overall'],
}

SUFFIXES = {
    'HallusionBench': "score",
    'OCRBench': 'score',
    'COCO_VAL': 'score',
    'MathVision_MINI': 'gpt-4o-mini_score',
    'MathVista_MINI': 'gpt-4o-mini_score'
}

EXTENSIONS = {
    "OCRBench": "json",
    'COCO_VAL': "json"
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./outputs")
    parser.add_argument('--summary_dir', type=str, default='./summaries')
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--precision', type=int, default=4)
    return parser.parse_args()


def float_convert(df):
    for col in df.columns:
        try:
            df[col] = df[col].float()
        except:
            pass
    return df

def json_to_csv(path):
    with open(path, 'r') as file_ptr:
        ddict = json.load(file_ptr)
    for k, v in ddict.items():
        if not isinstance(v, list): ddict[k] = [v]
    df = pd.DataFrame.from_dict(ddict)
    return df

def get_ckpt_from_path(path: str):
    splits = path.split("<ckpt>")
    return splits[1]


def main():
    args = parse_args()

    # convenience vars
    base_folder = args.input_dir
    models = list([m for m in args.model]) if args.model else list(supported_VLM.keys())
    datasets = list([d for d in args.data]) if args.data else SUPPORTED_DATASETS

    # loop through model / dataset combos
    for dataset in datasets:
        dataset_dfs = []
        for model in models:
            suffix = SUFFIXES.get(dataset, 'acc')
            ext = EXTENSIONS.get(dataset, 'csv')

            # standard evaluation with vlmevalkit models
            fpath = osp.join(base_folder, model, f"{model}_{dataset}_{suffix}.{ext}")
            if not osp.exists(fpath): 
                print(f"BEWARE! {model} x {dataset} combinations doesn't exist!")
            else:
                # inject ds and model name for easier visualization
                if ext == "json":
                    df = json_to_csv(fpath)
                else:
                    df = pd.read_csv(fpath)
                df['dataset'] = [dataset] * len(df)
                df['model'] = [model] * len(df)
                df['ckpt'] = ['source'] * len(df)
                dataset_dfs.append(df)
            
            # within each model folder, we're supposed to have a list of files 
            # reporting eval results for each ckpt
            ckpt_files = glob(osp.join(base_folder, model, f"{model}_<ckpt>*<ckpt>_{dataset}_{suffix}.{ext}"))
            if len(ckpt_files) == 0:
                print(f"Looks like no evals with DMT checkpoints are present for {model} x {dataset}. Please check.")
            else:
                # inject ds and model name for easier visualization
                for ckpt_eval in ckpt_files:
                    if ext == "json":
                        df = json_to_csv(ckpt_eval)
                    else:
                        df = pd.read_csv(ckpt_eval)
                    df['dataset'] = [dataset] * len(df)
                    df['model'] = [model] * len(df)
                    df['ckpt'] = [get_ckpt_from_path(ckpt_eval)] * len(df)
                    dataset_dfs.append(df)  
        
        # merge into one
        if not len(dataset_dfs): 
            print(f"No experiments found for {dataset}.")
            continue
        else:
            print(f"Found {len(dataset_dfs)} experiment(s) for {dataset}. Summarizing and sorting...")
        
        dataset_summary = pd.concat(dataset_dfs)
        dataset_summary = float_convert(dataset_summary).round(args.precision)
        dataset_summary = dataset_summary.sort_values(by=['dataset', 'model']+SORTING_KEYS[dataset], ascending=False)
        
        # re-order columns
        existing_cols, custom_cols = list(dataset_summary.columns[:-3]), list(dataset_summary.columns[-3:])
        custom_cols.extend(existing_cols)
        dataset_summary = dataset_summary[custom_cols]

        # dump into dataset-specific folders
        outpath = osp.join(args.summary_dir, dataset, "summary.tsv")
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        dataset_summary.to_csv(outpath, index=False, sep="\t")

        print(f"Saved {dataset} summary at {outpath}.", end='\n'+'='*50+'\n')


if __name__ == "__main__":
    main()
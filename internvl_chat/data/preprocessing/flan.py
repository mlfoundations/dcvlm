###################################
# 1. download the FLAN dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil


# path where parquets are temporarily stored
DOWNLOAD_DIR = "flan_parquets"
# output dir
OUTPUT_DIR = "./flan"


def main():
    snapshot_download(repo_id="Open-Orca/FLAN", repo_type="dataset", local_dir=DOWNLOAD_DIR)

    # recursively iterate all directories to retrieve all .parquet files
    parquets = [
        os.path.join(root, x)
        for root, dirs, files in os.walk(DOWNLOAD_DIR)
        for x in files
        if x.endswith(".parquet")
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # iterate over parquets
    for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
        df = pd.read_parquet(parquet)

        # jsonl for each parquet
        output_file = os.path.join(OUTPUT_DIR, f"preprocessed_{parquet_index}.jsonl")
        cnt = -1  # init count for each jsonl

        new_jsonl_files = []

        # iterate over parquet rows
        for sample_index, row in df.iterrows():
            cnt += 1

            # FLAN is text-only, thus retrieve inputs and targets
            conv = [
                {"from": "human", "value": row["inputs"]},
                {"from": "gpt", "value": row["targets"]},
            ]
            entry = {"id": cnt, "conversations": conv}

            new_jsonl_files.append(entry)

        # dump immediately each entry as the size can quickly saturate RAM
        with open(output_file, "w") as fp:
            for jsonl in new_jsonl_files:
                fp.write(json.dumps(jsonl) + "\n")

        print(
            f"Preprocessing complete! Saved as '{output_file}' with 0 saved images in {OUTPUT_DIR} and {cnt+1} total samples."
        )
    shutil.rmtree(DOWNLOAD_DIR)


if __name__ == "__main__":
    main()

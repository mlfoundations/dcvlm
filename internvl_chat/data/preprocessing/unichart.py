"""
Add UniChart dataset
https://huggingface.co/datasets/ahmed-masry/unichart-pretrain-data
"""

import os
import re
import random
from huggingface_hub import snapshot_download
from tqdm import tqdm
import json
import zipfile
import shutil
from tqdm import tqdm
import pandas as pd

random.seed(0)

OUTPUT_DIR = "./unichart"

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
snapshot_download(
    repo_id="ahmed-masry/UniChart-pretrain-images",
    repo_type="dataset",
    allow_patterns=["UniChart Images.zip"],
    local_dir=f"{OUTPUT_DIR}/tmp",
)
# unzip UniChart Images.zip
print("Extracting images")
with zipfile.ZipFile(f"{OUTPUT_DIR}/tmp/UniChart Images.zip") as images_zip:
    file_names = sorted(
        filter(lambda file: file.endswith(".png"), images_zip.namelist())
    )
    images_zip.extractall(f"{OUTPUT_DIR}/tmp")

for i, file_name in tqdm(
    enumerate(file_names), desc="Moving images", total=len(file_names)
):
    fp = os.path.join(OUTPUT_DIR, "tmp", file_name)
    shutil.move(fp, os.path.join(OUTPUT_DIR, "images", os.path.basename(file_name)))

snapshot_download(
    repo_id="ahmed-masry/unichart-pretrain-data",
    repo_type="dataset",
    allow_patterns=["data/*.parquet"],  # Only download files in the data folder
    local_dir=f"{OUTPUT_DIR}/image_parquets",
)

query_patterns_to_remove = "<opencqa>|<chartqa>|<summarize_chart>|<extract_data_table>|<extract_data_values>"
preprocessed, cnt = [], 0
for file_name in sorted(os.listdir(f"{OUTPUT_DIR}/image_parquets/data")):
    print(f"Processing {file_name}")
    p = pd.read_parquet(f"{OUTPUT_DIR}/image_parquets/data/{file_name}")
    for i, entry in tqdm(p.iterrows(), desc=f"Processing rows", total=len(p)):
        query = re.sub(query_patterns_to_remove, "", entry["query"]).strip()
        # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        image_beginning = random.choice([0, 1])
        value_str = f"<image>{query}" if image_beginning else f"{query}<image>"
        convs = [
            {"from": "human", "value": value_str},
            {"from": "gpt", "value": entry["label"]},
        ]
        assert os.path.exists(
            os.path.join(OUTPUT_DIR, "images", entry["imgname"])
        ), "Image in dataset does not exist on filesystem."
        preprocessed.append({
            "id": cnt,
            "image": os.path.join("images", entry["imgname"]),
            "conversations": convs,
        })
        cnt += 1

shutil.rmtree(f"{OUTPUT_DIR}/tmp")
shutil.rmtree(f"{OUTPUT_DIR}/image_parquets")

with open(f"{OUTPUT_DIR}/preprocessed.jsonl", "w") as f:
    for entry in tqdm(preprocessed, desc="Writing to jsonl", total=len(preprocessed)):
        f.write(json.dumps(entry) + "\n")

print(f"Downloaded UniChart dataset to {OUTPUT_DIR}")

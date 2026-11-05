"""
ST-VQA dataset
https://huggingface.co/datasets/vikhyatk/st-vqa
"""

import os
import random
from huggingface_hub import snapshot_download
from tqdm import tqdm
import json
import shutil
from tqdm import tqdm
import pandas as pd

random.seed(0)

OUTPUT_DIR = "./st-vqa"

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
snapshot_download(
    repo_id="vikhyatk/st-vqa",
    repo_type="dataset",
    allow_patterns=["data/*.parquet"],  # Only download files in the data folder
    local_dir=f"{OUTPUT_DIR}/image_parquets",
)

preprocessed, cnt = [], 0
for file_name in sorted(os.listdir(f"{OUTPUT_DIR}/image_parquets/data")):
    print(f"Processing {file_name}")
    p = pd.read_parquet(f"{OUTPUT_DIR}/image_parquets/data/{file_name}")
    for i, entry in tqdm(p.iterrows(), desc=f"Processing rows", total=len(p)):
        image_path = os.path.join("images", entry["image"]["path"])
        with open(os.path.join(OUTPUT_DIR, image_path), "wb") as f:
            f.write(entry["image"]["bytes"])
        assert os.path.exists(
            os.path.join(OUTPUT_DIR, image_path)
        ), "Image in dataset does not exist on filesystem."
        for qa in entry["qas"]:
            query = qa["question"]
            # Always pick the first answer as it seems to be the more accurate one
            answer = qa["answers"][0]
            # answer = random.choice(qa["answers"])
            # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            image_beginning = random.choice([0, 1])
            value_str = f"<image>{query}" if image_beginning else f"{query}<image>"
            convs = [
                {"from": "human", "value": value_str},
                {"from": "gpt", "value": answer},
            ]
            preprocessed.append({
                "id": cnt,
                "image": image_path,
                "conversations": convs,
            })
            cnt += 1


with open(f"{OUTPUT_DIR}/preprocessed.jsonl", "w") as f:
    for entry in tqdm(preprocessed, desc="Writing to jsonl", total=len(preprocessed)):
        f.write(json.dumps(entry) + "\n")

shutil.rmtree(f"{OUTPUT_DIR}/image_parquets")
print(f"Downloaded ST-VQA dataset to {OUTPUT_DIR}")

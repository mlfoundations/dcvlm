from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="HuggingFaceH4/no_robots",
    repo_type="dataset",
    allow_patterns=["data/train_sft*"],
    local_dir="no_robots_data"
)

output_dir = "./no_robots"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquet_files = [x for x in os.listdir("./no_robots_data/data") if x.endswith(".parquet")]
for parquet_file in tqdm(parquet_files, ascii=True):
    df = pd.read_parquet(os.path.join("./no_robots_data", "data", parquet_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        messages = row["messages"]
        conversations = []

        for msg in messages:
            if msg["role"] == "user":
                conversations.append({"from":"human", "value":msg["content"]})
            else:
                conversations.append({"from":"gpt", "value":msg["content"]})

        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./no_robots/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('no_robots_data')

print("Preprocessing complete! Saved as 'no_robots/preprocessed.jsonl' with {} samples.".format(n_samples))
from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="HuggingFaceH4/Code-Feedback",
    repo_type="dataset",
    allow_patterns=["data/train*"],
    local_dir="cf_data"
)

output_dir = "./code-feedback"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquet_files = [x for x in os.listdir("./cf_data/data") if x.endswith(".parquet")]
for parquet_file in tqdm(parquet_files, ascii=True):
    df = pd.read_parquet(os.path.join("./cf_data", "data", parquet_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        conversations = list(row["messages"])

        # Rename rows
        for msg in conversations:
            if msg["role"] == "user":
                msg["from"] = "human"
            else:
                msg["from"] = "gpt"
            
            msg["value"] = msg.pop("content")

        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./code-feedback/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('cf_data')

print("Preprocessing complete! Saved as 'code-feedback/preprocessed.jsonl' with {} samples.".format(n_samples))

from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="HuggingFaceH4/ultrachat_200k",
    repo_type="dataset",
    allow_patterns=["data/train*"],
    local_dir="uc_data"
)

output_dir = "./ultrachat_200k"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquet_files = [x for x in os.listdir("./uc_data/data") if x.endswith(".parquet")]
for parquet_file in tqdm(parquet_files, ascii=True):
    df = pd.read_parquet(os.path.join("./uc_data", "data", parquet_file))
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

with open('./ultrachat_200k/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('uc_data')

print("Preprocessing complete! Saved as 'ultrachat_200k/preprocessed.jsonl' with {} samples.".format(n_samples))
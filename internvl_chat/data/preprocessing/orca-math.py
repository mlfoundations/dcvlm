from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="microsoft/orca-math-word-problems-200k",
    repo_type="dataset",
    allow_patterns=["data/train*"],
    local_dir="orca_math_data"
)

output_dir = "./orca-math"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquet_files = [x for x in os.listdir("./orca_math_data/data") if x.endswith(".parquet")]
for parquet_file in tqdm(parquet_files, ascii=True):
    df = pd.read_parquet(os.path.join("./orca_math_data", "data", parquet_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        question = row["question"]
        answer = row["answer"]

        conversations = []
        conversations.append({"from":"human", "value":question})
        conversations.append({"from":"gpt", "value":answer})

        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('orca_math_data')

print("Preprocessing complete! Saved as 'orca-math/preprocessed.jsonl' with {} samples.".format(n_samples))
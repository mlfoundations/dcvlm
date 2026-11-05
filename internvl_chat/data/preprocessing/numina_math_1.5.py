from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="AI-MO/NuminaMath-1.5",
    repo_type="dataset",
    allow_patterns=["data/train*"],
    local_dir="nm_data"
)

output_dir = "./numina_math_1.5"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquet_files = [x for x in os.listdir("./nm_data/data") if x.endswith(".parquet")]
for parquet_file in tqdm(parquet_files, ascii=True):
    df = pd.read_parquet(os.path.join("./nm_data", "data", parquet_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        conversations = [{"from":"human", "value":row["problem"]}, {"from":"gpt", "value":row["solution"]}]
            
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./numina_math_1.5/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('nm_data')

print("Preprocessing complete! Saved as 'numina_math_1.5/preprocessed.jsonl' with {} samples.".format(n_samples))
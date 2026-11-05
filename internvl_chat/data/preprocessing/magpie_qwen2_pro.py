from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="Magpie-Align/Magpie-Qwen2-Pro-1M-v0.1",
    repo_type="dataset",
    local_dir="magpie_data"
)

output_dir = "./magpie_qwen2_pro"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquet_files = [x for x in os.listdir("./magpie_data/data") if x.endswith(".parquet")]
for parquet_file in tqdm(parquet_files, ascii=True):
    df = pd.read_parquet(os.path.join("./magpie_data", "data", parquet_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        conversations = list(row["conversations"])
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./magpie_qwen2_pro/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('magpie_data')

print("Preprocessing complete! Saved as 'magpie_qwen2_pro/preprocessed.jsonl' with {} samples.".format(n_samples))
from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="banksy235/XCoder-80K",
    repo_type="dataset",
    allow_patterns=["data/train*"],
    local_dir="xcoder_data"
)

output_dir = "./xcoder_80k"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquet_files = [x for x in os.listdir("./xcoder_data/data") if x.endswith(".parquet")]
for parquet_file in tqdm(parquet_files, ascii=True):
    df = pd.read_parquet(os.path.join("./xcoder_data", "data", parquet_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        conversations = list(row["messages"])

        # Rename rows
        for msg in conversations:
            if msg["role"] == "user":
                msg["from"] = "human"
            else:
                msg["from"] = "gpt"
            
            # Remove the role key
            msg.pop("role")
            # Rename content to value
            msg["value"] = msg.pop("content")

        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./xcoder_80k/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('xcoder_data')

print("Preprocessing complete! Saved as 'xcoder_80k/preprocessed.jsonl' with {} samples.".format(n_samples))
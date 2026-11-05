from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="openbmb/UltraFeedback",
    repo_type="dataset",
    local_dir="uf_data"
)

output_dir = "./ultrafeedback"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_prompts = 0
n_samples = 0

jsonl_files = [x for x in os.listdir("./uf_data") if x.endswith(".jsonl")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./uf_data", jsonl_file), lines=True)
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        n_prompts += 1
        completions = row["completions"]
        for completion in completions:
            conversations = [{"from":"human", "value":row["instruction"]}, {"from":"gpt", "value":completion["response"]}]
            entry_ = {"id":n_samples, "conversations": conversations}
            new_jsonl_entries.append(entry_)
            n_samples += 1

with open('./ultrafeedback/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('uf_data')

print("Preprocessing complete! Saved as 'ultrafeedback/preprocessed.jsonl' with {} prompts and {} total samples.".format(n_prompts, n_samples))
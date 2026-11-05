from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="Yukang/LongAlpaca-12k",
    repo_type="dataset",
    local_dir="la_data"
)

output_dir = "./long_alpaca"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./la_data") if x.endswith(".json")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./la_data", jsonl_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        conversations = [{"from":"human", "value":row["instruction"]}, {"from":"gpt", "value":row["output"]}]
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./long_alpaca/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('la_data')

print("Preprocessing complete! Saved as 'long_alpaca/preprocessed.jsonl' with {} total samples.".format(n_samples))
from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="THUDM/LongCite-45k",
    repo_type="dataset",
    local_dir="lc_data"
)

output_dir = "./long-cite-45k"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./lc_data") if x.endswith(".jsonl")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./lc_data", jsonl_file), lines=True)
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        conversations = [{"from":"human", "value":row["prompt"]}, {"from":"gpt", "value":row["response"]}]
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./long-cite-45k/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('lc_data')

print("Preprocessing complete! Saved as 'long-cite-45k/preprocessed.jsonl' with {} total samples.".format(n_samples))
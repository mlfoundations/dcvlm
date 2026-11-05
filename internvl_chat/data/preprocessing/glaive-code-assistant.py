from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import re

snapshot_download(
    repo_id="glaiveai/glaive-code-assistant",
    repo_type="dataset",
    local_dir="glaive_data"
)

output_dir = "./glaive-code-assistant"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./glaive_data") if x.endswith(".json")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./glaive_data", jsonl_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True, total=len(df)):
        conversations = [{"from":"human", "value":row["question"]}, {"from":"gpt", "value":row["answer"]}]
        
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./glaive-code-assistant/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('glaive_data')

print("Preprocessing complete! Saved as 'glaive-code-assistant/preprocessed.jsonl' with {} total samples.".format(n_samples))
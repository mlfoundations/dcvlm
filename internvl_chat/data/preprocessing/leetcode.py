from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="RayBernard/leetcode",
    repo_type="dataset",
    local_dir="lc_data"
)

output_dir = "./leetcode"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./lc_data") if x.endswith(".jsonl")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./lc_data", jsonl_file), lines=True)
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        text = row["text"]
        human_text, gpt_text = text.split("### Response:\n")
        conversations = [{"from":"human", "value":human_text}, {"from":"gpt", "value":gpt_text}]
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./leetcode/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('lc_data')

print("Preprocessing complete! Saved as 'leetcode/preprocessed.jsonl' with {} total samples.".format(n_samples))
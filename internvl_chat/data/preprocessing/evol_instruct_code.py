from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="nickrosh/Evol-Instruct-Code-80k-v1",
    repo_type="dataset",
    local_dir="evol_data"
)

output_dir = "./evol_instruct_code"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

json_files = [x for x in os.listdir("./evol_data") if x.endswith(".json")]
for json_file in tqdm(json_files, ascii=True):
    df = pd.read_json(os.path.join("./evol_data", json_file))

    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        conversations = [{"from":"human", "value":row["instruction"]}, {"from":"gpt", "value":row["output"]}]
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./evol_instruct_code/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('evol_data')

print("Preprocessing complete! Saved as 'evol_instruct_code/preprocessed.jsonl' with {} samples.".format(n_samples))
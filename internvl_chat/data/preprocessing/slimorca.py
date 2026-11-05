from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="Open-Orca/SlimOrca",
    repo_type="dataset",
    local_dir="slimorca_data"
)

output_dir = "./slimorca"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

json_files = [x for x in os.listdir("./slimorca_data") if x.endswith(".jsonl")]
for json_file in tqdm(json_files, ascii=True):

    contents = [json.loads(xx) for xx in open(os.path.join("./slimorca_data", json_file))]

    for sample_index, row in tqdm(enumerate(contents), total=len(contents), ascii=True):

        convs = []
        has_sys_prompt = False

        for cv in row["conversations"]:
            # remove system prompt
            if cv["from"] not in ["human", "gpt"]:
                has_sys_prompt = True
                continue

            # keep only "from" and "value" keys
            convs.append({"from": cv["from"], "value": cv["value"]})


        if has_sys_prompt:
            assert len(convs) == len(row["conversations"]) - 1
        else:
            assert len(convs) == len(row["conversations"])

        entry_ = {"id":n_samples, "conversations": convs}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./slimorca/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('slimorca_data')

print("Preprocessing complete! Saved as 'slimorca/preprocessed.jsonl' with {} samples.".format(n_samples))
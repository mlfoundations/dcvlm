from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="YeungNLP/firefly-train-1.1M",
    repo_type="dataset",
    local_dir="firefly_data"
)

output_dir = "./firefly"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

json_files = [x for x in os.listdir("./firefly_data") if x.endswith(".jsonl")]
for json_file in tqdm(json_files, ascii=True):

    contents = [json.loads(xx) for xx in open(os.path.join("firefly_data", json_file)).readlines()]

    for sample_index, row in tqdm(enumerate(contents), ascii=True, total=len(contents)):

        assert len(list(row.keys())) == 3, "sample has more than three keys!!!"

        conversations = [{"from":"human", "value":row["input"]}, {"from":"gpt", "value":row["target"]}]
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./firefly/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('firefly_data')

print("Preprocessing complete! Saved as 'firefly/preprocessed.jsonl' with {} samples.".format(n_samples))
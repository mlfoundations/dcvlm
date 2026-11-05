from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="knowrohit07/know-saraswati-cot",
    repo_type="dataset",
    local_dir="know_saraswati_cot_data"
)

output_dir = "./know-saraswati-cot"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

json_files = [x for x in os.listdir("./know_saraswati_cot_data") if x.endswith(".json")]
for json_file in tqdm(json_files, ascii=True):
    df = pd.read_json(os.path.join("./know_saraswati_cot_data", json_file))

    for sample_index, row in tqdm(df.iterrows(), ascii=True):

        assert len(list(row.keys())) == 2, "sample has more than two keys!!!"

        conversations = [{"from":"human", "value":row["instruction"]}, {"from":"gpt", "value":row["output"]}]
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./know-saraswati-cot/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('know_saraswati_cot_data')

print("Preprocessing complete! Saved as 'know-saraswati-cot/preprocessed.jsonl' with {} samples.".format(n_samples))
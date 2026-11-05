import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import requests
from zipfile import ZipFile

output_dir = "./unnatural_instructions"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("./ui_data", exist_ok=True)

url = "https://github.com/orhonovich/unnatural-instructions/raw/main/data/core_data.zip"
local_zip_path = "./ui_data/core_data.zip"

response = requests.get(url)
with open(local_zip_path, 'wb') as f:
    f.write(response.content)

with ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall("./ui_data")

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./ui_data") if x.endswith(".jsonl")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./ui_data", jsonl_file), lines=True)
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        instances = row["instances"]
        for instance in instances:
            conversations = [{"from":"human", "value":instance["instruction_with_input"]}, {"from":"gpt", "value":instance["output"]}]
            entry_ = {"id":n_samples, "conversations": conversations}
            new_jsonl_entries.append(entry_)
            n_samples += 1

with open('./unnatural_instructions/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('ui_data')

print("Preprocessing complete! Saved as 'unnatural_instructions/preprocessed.jsonl' with {} total samples.".format(n_samples))
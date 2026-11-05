from huggingface_hub import snapshot_download
import pandas as pd
import os
import zipfile
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="yuyijiong/Long-Instruction-with-Paraphrasing",
    repo_type="dataset",
    allow_patterns=["long_instruction_paraphrasing.zip"],
    local_dir="lip_data"
)

output_dir = "./long-instruction-with-paraphrasing"
os.makedirs(output_dir, exist_ok=True)

def read_zip(zip_path, n_samples=0):
    zip_entries = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_names = z.namelist()  # List files in the zip
        
        for file_name in file_names:
            if file_name.endswith(".json"):  # Adjust for your file format
                with z.open(file_name) as f:
                    df = pd.read_json(f, lines=True)
                    for sample_index, row in tqdm(df.iterrows(), ascii=True):
                        messages = row["messages"]
                        conversations = []

                        for msg in messages:
                            if msg["role"] == "user":
                                conversations.append({"from":"human", "value":msg["content"]})
                            else:
                                conversations.append({"from":"gpt", "value":msg["content"]})

                        entry_ = {"id":n_samples, "conversations": conversations}
                        zip_entries.append(entry_)
                        n_samples += 1

    return zip_entries

new_jsonl_entries = []
n_samples = 0

zip_files = [x for x in os.listdir("./lip_data") if x.endswith(".zip")]
for zip_file in tqdm(zip_files, ascii=True):
    zip_path = os.path.join("./lip_data", zip_file)
    new_jsonl_entries.extend(read_zip(zip_path, n_samples))
    n_samples += len(new_jsonl_entries)

with open('./long-instruction-with-paraphrasing/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('lip_data')

print("Preprocessing complete! Saved as 'long-instruction-with-paraphrasing/preprocessed.jsonl' with {} samples.".format(n_samples))
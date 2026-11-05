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
    repo_id="fnlp/moss-002-sft-data",
    repo_type="dataset",
    allow_patterns=["en*"],
    local_dir="moss_data"
)

output_dir = "./moss"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./moss_data") if x.endswith(".json")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./moss_data", jsonl_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True, total=len(df)):
        text = row["plain_text"]
        conversations = []

        # Split the text into speaker-message pairs
        pattern = re.findall(r'\[(Human|MOSS)\]: (.*?)(?=(?:\[Human\]|\[MOSS\]|$))', text, re.DOTALL)
    
        for speaker, message in pattern:
            # Remove <eoh> and <eoa> tokens
            message = message.replace('<eoh>', '').replace('<eoa>', '').strip()
            
            conversations.append({
                "from": "human" if speaker == "Human" else "gpt",
                "value": message
            })
        
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./moss/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('moss_data')

print("Preprocessing complete! Saved as 'moss/preprocessed.jsonl' with {} total samples.".format(n_samples))
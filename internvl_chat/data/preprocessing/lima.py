###################################
# 1. download the lima dataset using the snapshot_download below !!!ENSURE YOU SET HF_TOKEN
# 2. run this script from the same folder to get the preprocessed version!
###################################

import json
from huggingface_hub import snapshot_download
import os

if 'HF_TOKEN' not in os.environ:
    raise ValueError('Please set your HF_TOKEN environment variable. If you dont have one, create one at: https://huggingface.co/settings/tokens and do export HF_TOKEN=hf_XXXXXX.')

output_dir = "./lima"
os.makedirs(output_dir, exist_ok=True)

snapshot_download(
    repo_id="GAIR/lima",
    repo_type="dataset",
    allow_patterns=["train.jsonl"],  # Only download train file
    local_dir="./lima",
    token=os.environ['HF_TOKEN']
)

os.rename('./lima/train.jsonl', './lima/lima_train.jsonl')

with open('./lima/lima_train.jsonl', 'r') as f:
    dataset = [json.loads(x) for x in f.readlines()]

final = []
for idx, item in enumerate(dataset):
    cons = []
    for t_idx, t in enumerate(item['conversations']):
        pre = 'human' if t_idx%2==0 else 'gpt'
        cons.append({"from":pre, "value":t})
    preprocessed = {"id":idx, "conversations":cons}
    final.append(preprocessed)

with open('./lima/preprocessed.jsonl', 'w') as f:
    for entry in final:
        f.write(json.dumps(entry) + '\n')

os.remove('./lima/lima_train.jsonl')

print("Preprocessing complete! Saved as 'lima/preprocessed.jsonl'.")
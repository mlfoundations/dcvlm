# Use hf_transfer for faster download
# https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads

from huggingface_hub import snapshot_download
from pathlib import Path
from tqdm import tqdm
import random
import json
import shutil

random.seed(0)

download_dir = Path(__file__).parent / "metamathqa_download"
out_dir = Path(__file__).parent / "metamathqa"
out_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="meta-math/MetaMathQA",
    repo_type="dataset",
    allow_patterns=["*.json"],
    local_dir=download_dir,
)

entries = []
json_files = list(download_dir.glob("*.json"))
for i, json_file in enumerate(json_files):
    with open(json_file, "r") as f:
        data = json.load(f)

    for row in tqdm(
        data,
        total=len(data),
        ascii=True,
        desc=f"Processing file {i+1}/{len(json_files)}...",
    ):
        conversations = [
            {"from": "human", "value": row["query"]},
            {"from": "gpt", "value": row["response"]},
        ]

        entry = {
            "id": len(entries),
            "conversations": conversations,
        }
        entries.append(entry)

with open(out_dir / "preprocessed.jsonl", "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

shutil.rmtree(download_dir)

print("Preprocessing complete!")
print(f"Saved {len(entries)} samples to {out_dir}/preprocessed.jsonl")

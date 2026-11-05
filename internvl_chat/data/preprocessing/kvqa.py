"""
KVQA: Knowledge-aware Visual Question Answering dataset.
https://malllabiisc.github.io/resources/kvqa/

Hint: The original dataset url is no longer available.
      Therefore a huggingface url is used to download the dataset.
      (https://huggingface.co/datasets/wusize/kvqa)
      The HF repo does not contain the original metadata but is has
      been uploaded to `Data/kvqa_metadata.json` in the shared google drive folder.

Please download the kvqa_metadata.json file from the google drive and put it in the KVQA folder.
https://drive.google.com/file/d/1TKqPF5apZT-aUrrbjR42x-7N1_UjvSwe/view?usp=share_link
"""

import os
import random
from huggingface_hub import snapshot_download
from tqdm import tqdm
import json
import shutil
import tarfile
from tqdm import tqdm
from glob import glob

random.seed(0)

OUTPUT_DIR = "./kvqa"

os.makedirs(OUTPUT_DIR, exist_ok=True)
snapshot_download(
    repo_id="wusize/kvqa",
    repo_type="dataset",
    local_dir=f"{OUTPUT_DIR}/tmp",
)
tar_split_files = sorted(glob(f"{OUTPUT_DIR}/tmp/*.tar.gz.*"))
output_file = f"{OUTPUT_DIR}/tmp/kvqa.tar.gz"
# Combine the split files
with open(output_file, "wb") as outfile:
    for split_file in tqdm(tar_split_files, desc="Combining split files"):
        with open(split_file, "rb") as infile:
            outfile.write(infile.read())

print("Extracting images")
with tarfile.open(output_file, "r:gz") as tar:
    tar.extractall(f"{OUTPUT_DIR}")
os.remove(output_file)
shutil.move(f"{OUTPUT_DIR}/KVQAimgs", f"{OUTPUT_DIR}/images")

assert os.path.exists(f"{OUTPUT_DIR}/kvqa_metadata.json"), "Please download the kvqa_metadata.json file from the google drive (https://drive.google.com/file/d/1TKqPF5apZT-aUrrbjR42x-7N1_UjvSwe/view?usp=share_link) and put it in the KVQA folder."
with open(f"{OUTPUT_DIR}/kvqa_metadata.json", "r") as f:
    data = json.load(f)

preprocessed = []
for key, entry in tqdm(data.items(), desc="Processing entries", total=len(data)):
    image_path = entry['imgPath'].replace('KVQAimgs/', 'images/')
    assert os.path.exists(os.path.join(OUTPUT_DIR, image_path)), "Image in dataset does not exist on filesystem."
    convs = []
    # Iterate through multi-turn questions
    for i, (question, answer) in enumerate(zip(entry['Questions'], entry['Answers'])):
        value_str = question
        # Add image to the first question
        if i == 0:
            image_beginning = random.choice([0, 1])
            value_str = f"<image>{question}" if image_beginning else f"{question}<image>"
        convs += [
            {"from": "human", "value": value_str},
            {"from": "gpt", "value": answer},
        ]
    preprocessed.append({
        "id": key,
        "image": image_path,
        "conversations": convs,
    })

with open(f"{OUTPUT_DIR}/preprocessed.jsonl", "w") as f:
    for entry in tqdm(preprocessed, desc="Writing to jsonl", total=len(preprocessed)):
        f.write(json.dumps(entry) + "\n")

shutil.rmtree(f"{OUTPUT_DIR}/tmp")
os.remove(f"{OUTPUT_DIR}/kvqa_metadata.json")

print(f"Downloaded KVQA dataset to {OUTPUT_DIR}")

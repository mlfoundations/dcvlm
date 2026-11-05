from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import tarfile

FULL_IMG_PREFIX = "data/yiming/data/llava_cot/extracted_images"

download_dir = "./visualwebinstruct_data"

snapshot_download(
    repo_id="TIGER-Lab/VisualWebInstruct",
    repo_type="dataset",
    allow_patterns=["imgs.tar.gz", "mixed_conversation.jsonl"],
    local_dir=download_dir,
    revision="1b0831f7280b906b1132f0acd7b7eb24b3b12ecb"
)

output_dir = "./visualwebinstruct"
os.makedirs(output_dir, exist_ok=True)

# Path to the tar file
tar_path = os.path.join(download_dir, "imgs.tar.gz")

# Extract the tar file into the output directory
print("Extracting images!")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=output_dir)

print(f"Extracted images to {output_dir}")

new_jsonl_entries = []

with open(os.path.join(download_dir, "mixed_conversation.jsonl")) as f:
    contents = [json.loads(xx) for xx in f.readlines()]

for sample_index, content in tqdm(enumerate(contents), ascii=True, total=len(contents)):

    img_paths = None
    if "image" in content:
        img_paths = content["image"]
        img_paths_final = [os.path.join(FULL_IMG_PREFIX, im) for im in img_paths]

        all_true = True
        for im in img_paths_final:
            if not os.path.exists(os.path.join(output_dir, im)):
                all_true = False

        imgs_ = img_paths_final[0] if len(img_paths) == 1 else img_paths_final

    if not all_true:
        continue

    if img_paths is not None:
        entry_ = {"id":len(new_jsonl_entries), "image": imgs_, "conversations": content["conversations"]}
    else:
        entry_ = {"id":len(new_jsonl_entries), "conversations": content["conversations"]}

    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(download_dir)

print("Preprocessing complete! Saved as 'visualwebinstruct/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
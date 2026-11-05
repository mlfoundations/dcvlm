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
import zipfile

output_dir = "./mminstruct"
os.makedirs(output_dir, exist_ok=True)

def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")

snapshot_download(
    repo_id="yuecao0119/MMInstruct-GPT4V",
    repo_type="dataset",
    allow_patterns=["images.zip"],
    local_dir="mminstruct"
)

snapshot_download(
    repo_id="yuecao0119/MMInstruct-GPT4V",
    repo_type="dataset",
    allow_patterns=["jsons_all/*"],
    local_dir="mminstruct"
)

unzip_file(os.path.join(output_dir, 'images.zip'), os.path.join(output_dir, 'images'))

####### first preprocess caption_cn.jsonl
new_jsonl_entries = []

with open(os.path.join(output_dir, "jsons_all", "caption_cn.jsonl")) as f:
    json_contents = [json.loads(xx) for xx in f.readlines()]

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    new_jsonl_entries.append({"id": content["id"], "image": os.path.join("images/images", content["image"]), "conversations": content["conversations"]})
    assert os.path.exists(os.path.join(output_dir, "images/images", content["image"])), "image does not exist!!!"

with open('./mminstruct/caption_cn_preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'mminstruct/caption_cn_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

####### second preprocess caption_en.jsonl
new_jsonl_entries = []

with open(os.path.join(output_dir, "jsons_all", "caption_en.jsonl")) as f:
    json_contents = [json.loads(xx) for xx in f.readlines()]

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    new_jsonl_entries.append({"id": content["id"], "image": os.path.join("images/images", content["image"]), "conversations": content["conversations"]})
    assert os.path.exists(os.path.join(output_dir, "images/images", content["image"])), "image does not exist!!!"

with open('./mminstruct/caption_en_preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'mminstruct/caption_en_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

####### third preprocess qa_en.jsonl
new_jsonl_entries = []

with open(os.path.join(output_dir, "jsons_all", "qa_en.jsonl")) as f:
    json_contents = [json.loads(xx) for xx in f.readlines()]

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    new_jsonl_entries.append({"id": content["id"], "image": os.path.join("images/images", content["image"]), "conversations": content["conversations"]})
    assert os.path.exists(os.path.join(output_dir, "images/images", content["image"])), "image does not exist!!!"

with open('./mminstruct/qa_en_preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'mminstruct/qa_en_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

shutil.rmtree("./mminstruct/jsons_all")
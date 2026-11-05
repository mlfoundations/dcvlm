import gdown
import zipfile
import os
import shutil
import random
import json
from tqdm import tqdm
random.seed(0) # to ensure reproducibility while creating the dataset jsonl

# Google Drive file links and corresponding output paths
files = {
    "pretrain_images": "https://drive.google.com/uc?id=1zWpqnAcaG_dUwkJJUvP9FH9zq__c-ODY",
    "finetune_images": "https://drive.google.com/uc?id=1Ms7OCjcFQ18Whmujszpc9bTp0Jy0Dye4",
    "pretrain_annotations": "https://drive.google.com/uc?id=1_GCHFwrPGjp-9tZlDBwVkdz-L1ymchKY",
    "finetune_annotations": "https://drive.google.com/uc?id=1NHO8lly6pUo-fdyOAyWeGiQJWRb9qggk",
}

# Define output directories
output_dir = './llavar'
os.makedirs(output_dir, exist_ok=True)
output_dirs = {
    "images": os.path.join(output_dir, "images"),
    "pretrain": os.path.join(output_dir, "images/pretrain"),
    "finetune": os.path.join(output_dir, "images/finetune"),
    "json_pretrain": os.path.join(output_dir, "pretrain.json"),
    "json_finetune": os.path.join(output_dir, "finetune.json"),
}

# Ensure directories exist
for path in output_dirs.values():
    if "json" not in path:
        os.makedirs(path, exist_ok=True)

# Function to download files
def download_file(url, output):
    print(f"Downloading {output}...")
    gdown.download(url, output, quiet=False)

# Function to extract zip files
def extract_zip(file_path, extract_to):
    print(f"Extracting {file_path} to {extract_to}...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# Download and extract image zip files
download_file(files["pretrain_images"], "pretrain_images.zip")
download_file(files["finetune_images"], "finetune_images.zip")

extract_zip("pretrain_images.zip", output_dirs["pretrain"])
extract_zip("finetune_images.zip", output_dirs["finetune"])

# Move annotations JSON files
download_file(files["pretrain_annotations"], output_dirs["json_pretrain"])
download_file(files["finetune_annotations"], output_dirs["json_finetune"])

# Clean up zip files
os.remove("pretrain_images.zip")
os.remove("finetune_images.zip")

print("Download and extraction complete!")

###### Now reformat and pack as preprocessed.jsonl: Pretrain

new_jsonl_entries = []

with open(os.path.join(output_dir, 'pretrain.json')) as f:
    json_contents = json.load(f)

for index, contents in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_path = contents['image']
    conversations = contents['conversations']

    assert conversations is not None, 'conversations is None!!!'
    assert image_path is not None, 'image_path is None!!!'

    # some of the annotations are actually for the original llava_595k dataset, filter those out!
    if not os.path.exists(os.path.join(output_dirs['pretrain'], image_path)):
        continue

    entry_ = {"id":len(new_jsonl_entries), "image": 'images/pretrain/'+image_path, "conversations": conversations}
    new_jsonl_entries.append(entry_)

with open('./llavar/pretrain422k_preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'llavar/pretrain422k_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

os.remove(os.path.join(output_dir, 'pretrain.json'))

###### Now reformat and pack as preprocessed.jsonl: Finetune

new_jsonl_entries = []

with open(os.path.join(output_dir, 'finetune.json')) as f:
    json_contents = json.load(f)

for index, contents in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_path = contents['image']
    conversations = contents['conversations']

    assert conversations is not None, 'conversations is None!!!'
    assert image_path is not None, 'image_path is None!!!'

    # some of the annotations are actually for the original llava_150k dataset, filter those out!
    if not os.path.exists(os.path.join(output_dirs['finetune'], image_path)):
        continue

    entry_ = {"id":len(new_jsonl_entries), "image": 'images/finetune/'+image_path, "conversations": conversations}
    new_jsonl_entries.append(entry_)

with open('./llavar/finetune20k_preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'llavar/finetune20k_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

os.remove(os.path.join(output_dir, 'finetune.json'))

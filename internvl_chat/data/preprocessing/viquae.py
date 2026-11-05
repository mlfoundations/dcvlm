import json
import os
import io
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import shutil
import tarfile
import requests

TRAIN_JSONL_URL = 'https://huggingface.co/datasets/PaulLerner/viquae_dataset/resolve/main/train.jsonl'
TRAIN_IMAGES_URL = 'https://huggingface.co/datasets/PaulLerner/viquae_images/resolve/main/images.tar.gz'

###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

###### Download the files and unzip

output_dir = "./viquae"
os.makedirs(output_dir, exist_ok=True)

# Download train.jsonl
train_jsonl_path = os.path.join(output_dir, "train.jsonl")
download_file(TRAIN_JSONL_URL, train_jsonl_path)

# Download and extract images.tar.gz
images_tar_path = os.path.join(output_dir, "images.tar.gz")
download_file(TRAIN_IMAGES_URL, images_tar_path)

# Extract the tar.gz file
with tarfile.open(images_tar_path, 'r:gz') as tar:
    tar.extractall(output_dir)
    print(f"Extracted images to {output_dir}")

# Optionally remove the tar file after extraction
os.remove(images_tar_path)

###### Read and create preprocessed jsonl file

with open(train_jsonl_path) as f:
    json_contents = [json.loads(x) for x in f.readlines()]

new_jsonl_entries = []

# add entry to new jsonl list
for t in json_contents:
    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format("Question: {}".format(t['input']))}, {"from":"gpt", "value":t['output']['original_answer'].title()}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format("Question: {}".format(t['input']))}, {"from":"gpt", "value":t['output']['original_answer'].title()}]
    entry_ = {"id":len(new_jsonl_entries), "image": 'images/'+t['image'], "conversations": convs}
    assert os.path.exists('viquae/images/'+t['image']), 'Image does not exist!!'
    new_jsonl_entries.append(entry_)

with open('./viquae/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove('./viquae/train.jsonl')

print("Preprocessing complete! Saved as 'viquae/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
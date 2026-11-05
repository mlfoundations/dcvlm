from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0)  # Ensure reproducibility
import json
import shutil
import zipfile
import requests

############ Utility functions

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")

############ Download the dataset
output_dir = "./allava"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+"/images", exist_ok=True)
os.makedirs(output_dir+"/images/vflan_images", exist_ok=True)
os.makedirs(output_dir+"/images/laion_images", exist_ok=True)

snapshot_download(
    repo_id="FreedomIntelligence/ALLaVA-4V",
    repo_type="dataset",
    allow_patterns=["allava_laion/*", "allava_text/*.json", "allava_vflan/*.json", "others/*.json"],
    local_dir=output_dir
)

print("Downloading VFLAN images!")
url_ = "https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/resolve/main/image_191-task_1k.zip"
download_file(url_, os.path.join(output_dir, "vflan_images.zip"))
unzip_file(os.path.join(output_dir, "vflan_images.zip"), os.path.join(output_dir, "images", "vflan_images"))

laion_images = sorted([x for x in os.listdir(os.path.join(output_dir, "allava_laion", "image_chunks")) if ".zip" in x])
for i in range(len(laion_images)):
    unzip_file(os.path.join(output_dir, "allava_laion", "image_chunks", laion_images[i]), os.path.join(output_dir, "images", "laion_images"))


############ Preprocess ALLaVA-Caption-LAION-4V

new_jsonl_entries = []

with open(os.path.join(output_dir, "allava_laion", "ALLaVA-Caption-LAION-4V.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_path = content["image"]
    final_img_path = os.path.join("images", "laion_images", "images", os.path.basename(image_path))
    if not os.path.exists(os.path.join(output_dir, final_img_path)):
        print("{} image does not exist!! Skipping!!".format(final_img_path))
        continue

    convs = content["conversations"]
    assert convs is not None

    entry_ = {"id":len(new_jsonl_entries), "image": final_img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "laion_caption_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'allava/laion_caption_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

############ Preprocess ALLaVA-Instruct-LAION-4V

new_jsonl_entries = []

with open(os.path.join(output_dir, "allava_laion", "ALLaVA-Instruct-LAION-4V.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_path = content["image"]
    final_img_path = os.path.join("images", "laion_images", "images", os.path.basename(image_path))
    if not os.path.exists(os.path.join(output_dir, final_img_path)):
        print("{} image does not exist!! Skipping!!".format(final_img_path))
        continue

    convs = content["conversations"]
    assert convs is not None

    entry_ = {"id":len(new_jsonl_entries), "image": final_img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "laion_instruct_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'allava/laion_instruct_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))



############ Preprocess ALLaVA-Caption-VFLAN-4V

new_jsonl_entries = []

with open(os.path.join(output_dir, "allava_vflan", "ALLaVA-Caption-VFLAN-4V.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_path = content["image"]
    final_img_path = os.path.join("images", "vflan_images", "images_191task_1k", os.path.basename(image_path))
    if not os.path.exists(os.path.join(output_dir, final_img_path)):
        print("{} image does not exist!! Skipping!!".format(final_img_path))
        continue

    convs = content["conversations"]
    assert convs is not None

    entry_ = {"id":len(new_jsonl_entries), "image": final_img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "vflan_caption_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'allava/vflan_caption_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))




############ Preprocess ALLaVA-Instruct-VFLAN-4V

new_jsonl_entries = []

with open(os.path.join(output_dir, "allava_vflan", "ALLaVA-Instruct-VFLAN-4V.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_path = content["image"]
    final_img_path = os.path.join("images", "vflan_images", "images_191task_1k", os.path.basename(image_path))
    if not os.path.exists(os.path.join(output_dir, final_img_path)):
        print("{} image does not exist!! Skipping!!".format(final_img_path))
        continue

    convs = content["conversations"]
    assert convs is not None

    entry_ = {"id":len(new_jsonl_entries), "image": final_img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "vflan_instruct_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'allava/vflan_instruct_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))



############ Preprocess Evol-Instruct-GPT4-Turbo-143K.json

new_jsonl_entries = []

with open(os.path.join(output_dir, "allava_text", "Evol-Instruct-GPT4-Turbo-143K.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    convs = content["conversations"]
    assert convs is not None

    entry_ = {"id":len(new_jsonl_entries), "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "evol_instruct_text_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'allava/evol_instruct_text_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))



############ Preprocess Instruct-LAION-4oiterative.json

new_jsonl_entries = []

with open(os.path.join(output_dir, "others", "Instruct-LAION-4oiterative.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):

    image_path = content["image"]
    final_img_path = os.path.join("images", "laion_images", "images", os.path.basename(image_path))
    if not os.path.exists(os.path.join(output_dir, final_img_path)):
        print("{} image does not exist!! Skipping!!".format(final_img_path))
        continue

    convs = content["conversations"]
    assert convs is not None

    entry_ = {"id":len(new_jsonl_entries), "image": final_img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "others_laion4oiterative_instruct_text_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'allava/others_laion4oiterative_instruct_text_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))



############ Preprocess Instruct-LAION-4v_gemini_claude-ensembled.json

new_jsonl_entries = []

with open(os.path.join(output_dir, "others", "Instruct-LAION-4v_gemini_claude-ensembled.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):

    image_path = content["image"]
    final_img_path = os.path.join("images", "laion_images", "images", os.path.basename(image_path))
    if not os.path.exists(os.path.join(output_dir, final_img_path)):
        print("{} image does not exist!! Skipping!!".format(final_img_path))
        continue

    convs = content["conversations"]
    assert convs is not None

    entry_ = {"id":len(new_jsonl_entries), "image": final_img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "others_laion4vgeminiclaudeensembled_instruct_text_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'allava/others_laion4vgeminiclaudeensembled_instruct_text_preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))


##### Remove unnecessary files
shutil.rmtree(os.path.join(output_dir, "allava_laion"))
shutil.rmtree(os.path.join(output_dir, "allava_text"))
shutil.rmtree(os.path.join(output_dir, "allava_vflan"))
shutil.rmtree(os.path.join(output_dir, "others"))
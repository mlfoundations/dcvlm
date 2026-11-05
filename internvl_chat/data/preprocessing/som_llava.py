from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import io
import random
import re

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import zipfile
import gdown
import requests
import tarfile

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'


def extract_file_id(url):
    if 'id=' in url:
        return url.split('id=')[1]
    elif '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    else:
        raise ValueError("Invalid Google Drive URL")


def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")


def extract_zip_files(download_dir):
    zip_files = [f for f in os.listdir(download_dir) if f.endswith('.zip')]

    for zip_file in zip_files:
        zip_path = os.path.join(download_dir, zip_file)
        print(f"Extracting {zip_file}...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            print(f"Extracted {zip_file}")
        except zipfile.BadZipFile:
            print(f"Failed to extract {zip_file}. It may be corrupted.")

        # os.remove(zip_path)


def randomise_image(conversations):
    for conv in conversations:
        if conv["from"] == "human" and "<image>" in conv["value"]:
            parts = conv["value"].split("<image>")
            text = parts[1].strip() if len(parts) > 1 else ""
            if random.choice([0, 1]):
                conv["value"] = f"<image>\n{text}"
            else:
                conv["value"] = f"{text}\n<image>"
    return conversations


download_dir = "./som_llava_zip"
output_dir = "./som_llava"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

snapshot_download(
    repo_id="zzxslp/SoM-LLaVA",
    repo_type="dataset",
    allow_patterns=["som_llava_mix695k.json"],
    local_dir=download_dir)

snapshot_download(
    repo_id="howard-hou/OCR-VQA",
    repo_type="dataset",
    allow_patterns=["data/train-*"],
    local_dir=download_dir)

vg1_url = "https://drive.google.com/file/d/1_59IEd2H5DL6-FNYPwc1WGF0EGJVCfiN/view?usp=sharing"
vg2_url = "https://drive.google.com/file/d/1dB3-BdrwU9qV53dJWbytCK3XhyBY2KkL/view?usp=sharing"
GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"

urls = ([vg1_url, "vg1.zip"], [vg2_url, "vg2.zip"], [GQA_IMAGES_URL, "gqa.zip"])
folders = ['VG_100K', 'VG_100K_2', "images", "textvqa"]

if all(os.path.exists(os.path.join(download_dir, folder)) for folder in folders):
    print("All required folders exist. No download needed.")
else:
    if not all(os.path.exists(os.path.join(download_dir, file)) for url, file in urls):
        for url in urls:
            if "drive.google.com" in url[0]:
                file_id = extract_file_id(url[0])
                gdown.download(f'https://drive.google.com/uc?id={file_id}', os.path.join(download_dir, url[1]),
                               quiet=False)
            else:
                download_file(url[0], os.path.join(download_dir, url[1]))
    extract_zip_files(download_dir)

    download_file('https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/textvqa.tar.gz',
                  os.path.join(download_dir, 'textvqa.tar.gz'))
    with tarfile.open(os.path.join(download_dir, 'textvqa.tar.gz'), 'r:gz') as tar:
        tar.extractall(download_dir)

new_jsonl_entries = []
cnt = 0
imgs_saved = 0
imgs_coco = 0
rest_count = 0
ocr_vqa_images = {}

with open(os.path.join(download_dir, "som_llava_mix695k.json"), 'r') as f:
    data = json.load(f)
print("Read json file with {} entries".format(len(data)))


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


for idx, row in enumerate(tqdm(data, ascii=True)):
    if "model" in row.keys() or row["image"] == "":
        continue
    else:
        image_path_old = row["image"]

    if 'coco' in image_path_old:
        image_path = image_path_old.replace("coco/", "images/")
        image_path = image_path.replace("train2017", "train2017/train2017")
        image_path = image_path.replace("som_", "")
        assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image_path)), 'Image {} does not exist!!!'.format(
            image_path)
        file_name = image_path
        imgs_coco += 1

    elif 'VG_100K' in image_path_old:
        image_path = image_path_old.replace('vg/', '')
        assert os.path.exists(os.path.join(download_dir, image_path)), 'Image {} does not exist!!!'.format(
            os.path.join(download_dir, image_path))
        file_name = os.path.join(output_dir, "images", f"vg_{idx}.jpg")
        shutil.copyfile(os.path.join(download_dir, image_path), file_name)
        imgs_saved += 1

    elif 'gqa' in image_path_old:
        image_path = image_path_old.replace('gqa/', '')
        assert os.path.exists(os.path.join(download_dir, image_path)), 'Image {} does not exist!!!'.format(
            os.path.join(download_dir, image_path))
        file_name = os.path.join(output_dir, "images", f"gqa_{idx}.jpg")
        shutil.copyfile(os.path.join(download_dir, image_path), file_name)
        imgs_saved += 1

    elif 'textvqa' in image_path_old:
        image_path = image_path_old
        assert os.path.exists(os.path.join(download_dir, image_path)), 'Image {} does not exist!!!'.format(
            os.path.join(download_dir, image_path))
        file_name = os.path.join(output_dir, "images", f"textvqa_{idx}.jpg")
        shutil.copyfile(os.path.join(download_dir, image_path), file_name)
        imgs_saved += 1

    elif 'ocr_vqa' in image_path_old:
        ocr_vqa_images[row["id"]] = row["conversations"]
        continue

    # add entry to new jsonl list
    conversations = randomise_image(row["conversations"])
    if "coco" in image_path_old:
        entry_ = {"id": cnt, "image": image_path, "conversations": conversations}
    else:
        entry_ = {"id": cnt, "image": "images/" + os.path.basename(file_name), "conversations": conversations}
    new_jsonl_entries.append(entry_)
    cnt += 1

# store the id
image_ids = idx + 1
columns_to_read = ["image_id", "image"]

parquets = [x for x in os.listdir(f"{download_dir}/data") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):

    # df = pd.read_parquet(os.path.join(f"{download_dir}/data", parquet))
    df = pd.read_parquet(os.path.join(f"{download_dir}/data", parquet), columns=columns_to_read)
    df_filtered = df[df["image_id"].isin(ocr_vqa_images)]

    records = df_filtered.to_dict(orient="records")
    for row in records:
        image_id = row["image_id"]
        image_bytes = row["image"]["bytes"]
        assert image_bytes is not None, "image bytes are None!!!"

        image = Image.open(io.BytesIO(image_bytes))
        image_path = f"{output_dir}/images/ocrvqa_{image_ids}.png"
        image.save(image_path)

        conversations = randomise_image(ocr_vqa_images[image_id])
        new_jsonl_entries.append(
            {"id": cnt, "image": "images/" + os.path.basename(image_path), "conversations": conversations})
        cnt += 1
        image_ids += 1
        imgs_saved += 1

print(f"Saved {imgs_saved} images")
with open(f'{output_dir}/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(download_dir)
print(
    f"Preprocessing complete! Saved {imgs_coco} images from COCO and {imgs_saved} images from the rest in '{output_dir}/preprocessed.jsonl' with {cnt} total samples.")
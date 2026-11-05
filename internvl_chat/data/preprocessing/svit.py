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
import subprocess
import requests
import tarfile
import gc
from concurrent.futures import ProcessPoolExecutor

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

def extract_zip_with_unzip(zip_path, extract_path):
    try:
        result = subprocess.run(["unzip", zip_path, "-d", extract_path], check=True, text=True, capture_output=True)
        print(f"Extracted {zip_path} successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract {zip_path}:\n{e.stderr}")

def extract_zip_files(download_dir):
    if download_dir == "./svit_tars/raw":
        zip_files = [f for f in os.listdir(download_dir) if f.endswith('.zip') and "images" in f]
    else:
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

def process_json_file(json_file, download_dir, output_dir, existing_ids):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    local_images_saved = 0
    local_images_coco = 0
    local_ocr_vqa_images = {}
    
    for row in tqdm(data, ascii=True):
        if "model" in row.keys() or row["image"] == "":
            continue
        if row["id"] in existing_ids:
            continue
        
        image_id = row["id"]
        cnt = len(existing_ids) + len(results)
        
        if "vg" in row["image"]:
            image_path = row["image"].replace("vg", "raw")
            image_path = os.path.join(download_dir, image_path)
            assert os.path.exists(image_path), f'Image {image_path} does not exist!!!'
            file_name = os.path.join(output_dir, "images", f"vg_sample_{cnt}.png")
            shutil.copyfile(image_path, file_name)
            local_images_saved += 1
            
            conversations = randomise_image(row["conversations"])
            entry_ = {"id": random.randint(10**9, 10**10 - 1), "image": f'images/vg_sample_{cnt}.png', "conversations": conversations}
            results.append((entry_, image_id))
            
        elif "coco" in row["image"]:
            image_path = row["image"].replace("coco/", "images/")
            image_path = image_path.replace("train2017", "train2017/train2017")
            image_path = image_path.replace("train2014", "train2014/train2014")
            coco_image_path = os.path.join(os.environ['COCO_DIR'], image_path)
            
            assert os.path.exists(coco_image_path), f'Image {image_path} does not exist!!!'
            local_images_coco += 1
            
            conversations = randomise_image(row["conversations"])
            entry_ = {"id": random.randint(10**9, 10**10 - 1), "image": image_path, "conversations": conversations}
            results.append((entry_, image_id))
            
        elif 'textvqa' in row["image"]:
            image_path = row["image"]
            image_path = os.path.join(download_dir, image_path)
            assert os.path.exists(image_path), f'Image {image_path} does not exist!!!'
            file_name = os.path.join(output_dir, "images", f"textvqa_sample_{cnt}.png")
            shutil.copyfile(image_path, file_name)
            local_images_saved += 1
            
            conversations = randomise_image(row["conversations"])
            entry_ = {"id": random.randint(10**9, 10**10 - 1), "image": f'images/textvqa_sample_{cnt}.png', "conversations": conversations}
            results.append((entry_, image_id))
            
        elif 'ocr_vqa' in row["image"]:
            local_ocr_vqa_images[row["id"]] = row["conversations"]
            
        elif 'gqa' in row["image"]:
            image_path = row["image"]
            image_path = os.path.join(download_dir, image_path)
            image_path = image_path.replace('gqa/', '')
            assert os.path.exists(image_path), f'Image {image_path} does not exist!!!'
            file_name = os.path.join(output_dir, "images", f"gqa_sample_{cnt}.png")
            shutil.copyfile(image_path, file_name)
            local_images_saved += 1
            
            conversations = randomise_image(row["conversations"])
            entry_ = {"id": random.randint(10**9, 10**10 - 1), "image": f'images/gqa_sample_{cnt}.png', "conversations": conversations}
            results.append((entry_, image_id))
    
    return results, local_images_saved, local_images_coco, local_ocr_vqa_images

def process_parquet_file(args):
    parquet_file, download_dir, output_dir, ocr_vqa_images, start_cnt = args
    
    columns_to_read = ["image_id", "image"]
    df = pd.read_parquet(os.path.join(f"{download_dir}/data", parquet_file), columns=columns_to_read)
    df_filtered = df[df["image_id"].isin(ocr_vqa_images)]
    
    results = []
    local_images_saved = 0
    cnt = start_cnt
    
    records = df_filtered.to_dict(orient="records")
    for row in records:
        image_id = row["image_id"]
        image_bytes = row["image"]["bytes"]
        assert image_bytes is not None, "image bytes are None!!!"
        
        image = Image.open(io.BytesIO(image_bytes))
        image_path = f"{output_dir}/images/ocrvqa_sample_{cnt}.png"
        image.save(image_path)
        
        conversations = randomise_image(ocr_vqa_images[image_id])
        entry = {"id": random.randint(10**9, 10**10 - 1), "image": f"images/ocrvqa_sample_{cnt}.png", "conversations": conversations}
        results.append(entry)
        cnt += 1
        local_images_saved += 1
    
    return results, local_images_saved

def main():
    assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX' 

    download_dir = "./svit_tars"
    output_dir = "./svit"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    snapshot_download(
        repo_id="BAAI/SVIT",
        repo_type="dataset",
        local_dir=download_dir,
    )

    snapshot_download(
        repo_id="howard-hou/OCR-VQA",
        repo_type="dataset",
        allow_patterns=["data/train-*"],
        local_dir=download_dir)

    download_file('https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/textvqa.tar.gz', os.path.join(download_dir, 'textvqa.tar.gz'))
    with tarfile.open(os.path.join(download_dir,'textvqa.tar.gz'), 'r:gz') as tar:
        tar.extractall(download_dir)


    GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    download_file(GQA_IMAGES_URL, os.path.join(download_dir, "gqa.zip"))
    
    extract_zip_files(os.path.join(download_dir,"format/llava-v1.5"))
    extract_zip_with_unzip(os.path.join(download_dir,"raw/annotations.zip"), os.path.join(download_dir,"raw"))
    extract_zip_files(os.path.join(download_dir,"raw"))
    extract_zip_files(os.path.join(download_dir))
    

    jsons_format = [os.path.join(download_dir, "format/llava-v1.5", x) for x in
                   os.listdir(os.path.join(download_dir, "format/llava-v1.5")) if x.endswith(".json")]
    jsons_raw = [os.path.join(download_dir, "raw", x) for x in 
                os.listdir(os.path.join(download_dir, "raw")) if x.endswith(".json")]
    jsons = jsons_format + jsons_raw
    
    cnt = 0
    ids = []
    images_saved = 0
    images_coco = 0
    ocr_vqa_images = {}
    new_jsonl_entries = []

    json_tasks = [(json_file, download_dir, output_dir, ids) for json_file in jsons]
    
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(
            executor.map(process_json_file, *zip(*json_tasks)), 
            total=len(json_tasks),
            desc="Processing JSON files"
        ))
    
    for result_list, local_saved, local_coco, local_ocr_vqa in results:
        for entry, image_id in result_list:
            new_jsonl_entries.append(entry)
            ids.append(image_id)
        
        images_saved += local_saved
        images_coco += local_coco
        ocr_vqa_images.update(local_ocr_vqa)
    
    cnt = len(new_jsonl_entries)
    print(f"Processed {len(ids)} unique images from JSON files")
    print(f"Images saved: {images_saved}, COCO images: {images_coco}")
    print(f"Number of OCR_VQA samples: {len(ocr_vqa_images)}")
    
    
    columns_to_read = ["image_id", "image"]
    parquets = [x for x in os.listdir(f"{download_dir}/data") if x.endswith(".parquet")]
    for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):

        df = pd.read_parquet(os.path.join(f"{download_dir}/data", parquet), columns=columns_to_read)
        df_filtered = df[df["image_id"].isin(ocr_vqa_images)]

        records = df_filtered.to_dict(orient="records")
        for row in records:
            image_id = row["image_id"]
            image_bytes = row["image"]["bytes"]
            assert image_bytes is not None, "image bytes are None!!!"

            image = Image.open(io.BytesIO(image_bytes))
            image_path = f"{output_dir}/images/ocrvqa_sample_{cnt}.png"
            image.save(image_path)

            conversations = randomise_image(ocr_vqa_images[image_id])
            new_jsonl_entries.append({"id": random.randint(10**9, 10**10 - 1), "image": f"images/ocrvqa_sample_{cnt}.png", "conversations": conversations})
            cnt += 1
            images_saved += 1

    print(f"Total images saved: {images_saved}")
    print(f"Total entries: {len(new_jsonl_entries)}")
    
    with open(f'{output_dir}/preprocessed.jsonl', 'w') as f:
        for entry in new_jsonl_entries:
            f.write(json.dumps(entry) + '\n')
    
    gc.collect()
    shutil.rmtree(download_dir)

if __name__ == "__main__":
    main()

import argparse
from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import requests
import shutil
import tarfile

# Get filename and remove extension
DS_NAME = os.path.basename(__file__).split(".")[0] 
# Temporary this for download
TEMP_DIR = f"./{DS_NAME}_temp"
# Final directory for the images
IMAGES_DIR = f"./{DS_NAME}/images"
# Final path for annotations
ANNOTATIONS_FILE = f"./{DS_NAME}/preprocessed.jsonl"
# Prompts from https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data V3Det dataset
# Since CTW only labels chinese characters and skip all the other object, I will specify chinese characters
# Response is in the format \n<ref>TARGET</ref><box>[[x1, y1, x2, y2]]</box>
TARGET = "chinese characters"
PROMPT_REQUEST = f"Please detect and label all {TARGET} in the following image and mark their positions."
# Template to generate response. Provide a list of bbox
PROMPT_RESPONSE_HEADER = f"Sure, I will detect and label {TARGET} in the image and mark their positions.\n"

def pos_only(val):
    return val if val > 0 else 0

def prompt_response(targets, bboxes):
    response = PROMPT_RESPONSE_HEADER
    for target, bbox in zip(targets, bboxes):
        response += f"\n<ref>{target}</ref><box>{bbox}</box>"
    return response

# !!! CTW is in format x, y, w, h, while V3DET accepts x1, y1, x2, y2 and they should be converted in int
def convert_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    bbox = (int(x), int(y), int(x+w), int(y+h))
    bbox_n = normalize_coordinates(bbox, img_width, img_height)
    return [[pos_only(bbox_n[0]), pos_only(bbox_n[1]), pos_only(bbox_n[2]), pos_only(bbox_n[3])]]

### Copied from: https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data
def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box

def download_file(url, output_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, 'wb') as file:
                shutil.copyfileobj(response.raw, file)
        print(f"Downloaded to {output_path}")
    except Exception as e:
        print(f"Error: {e}")



def extract_tar(file_path, extract_to="."):
    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted to {extract_to}")



def main():

    os.makedirs(TEMP_DIR, exist_ok=True)

    if os.path.exists(ANNOTATIONS_FILE):
        os.remove(ANNOTATIONS_FILE)

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Urls for images and annotations
    url_images = lambda x: f"https://cloud.tsinghua.edu.cn/d/6f1cfc516c324f41a07b/files/?p=%2Fctw-public%2Fimages-trainval%2Fctw-trainval-{x}-of-26.tar&dl=1"
    url_annotations = "https://cloud.tsinghua.edu.cn/d/6f1cfc516c324f41a07b/files/?p=%2Fctw-public%2Fctw-annotations.tar.gz&dl=1"

    # Download annotations
    download_file(url_annotations, f"{TEMP_DIR}/ctw-annotations.tar.gz")
    extract_tar(f"{TEMP_DIR}/ctw-annotations.tar.gz", TEMP_DIR)

    # Download images
    for i in range(1, 26):
        print(f"Downloading {i} split of 26 ...")
        if i < 10:
            i = f"0{i}"
        output_path = f"{TEMP_DIR}/ctw-trainval-{i}-of-26.tar"
        download_file(url_images(i), output_path)
        extract_tar(output_path, IMAGES_DIR)

    # Process annotation file
    with open(f"{TEMP_DIR}/train.jsonl", "r") as f:
        lines = [json.loads(line) for line in f]

    new_jsonl_entries = []

    print("Processing {} annotations...".format(len(lines)), flush=True)

    not_found_imgs = 0

    for idx, line in enumerate(tqdm(lines, ascii=True, total=len(lines))):
        
        # flatten annotations as they are grouped by line and convert bboxes to 3VDet format
        flat_annotations = [annot for annot_list in line["annotations"] for annot in annot_list]
        width = line["width"]
        height = line["height"]
        bboxes = [convert_bbox(annot["adjusted_bbox"], width, height) for annot in flat_annotations]
        targets = [annot["text"] for annot in flat_annotations]
        file_path = os.path.join(IMAGES_DIR, line["file_name"])

        # build prompt response
        response = prompt_response(targets, bboxes)

        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        
        if not os.path.exists(file_path):
            not_found_imgs += 1
            continue

        if image_beginning:
            conv = [{"from":"human", "value":"<image>{}".format(PROMPT_REQUEST)}, {"from":"gpt", "value":response}]
        else:
            conv = [{"from":"human", "value":"{}<image>".format(PROMPT_REQUEST)}, {"from":"gpt", "value":response}]

        entry_ = {"id": idx, "image": f"images/{os.path.basename(file_path)}", "height": height, "width": width, "conversations": conv}
        new_jsonl_entries.append(entry_)


    with open(ANNOTATIONS_FILE, 'a') as f:
        for entry in new_jsonl_entries:
            f.write(json.dumps(entry) + '\n')

    shutil.rmtree(TEMP_DIR)
    print("Preprocessing complete! Saved as `{}` with {} not-saved images in {} and {} total samples.".format(ANNOTATIONS_FILE, not_found_imgs, IMAGES_DIR, len(lines)), flush=True)

if __name__ == "__main__":
    main()

        


    


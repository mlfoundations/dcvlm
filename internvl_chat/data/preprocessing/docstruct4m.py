###################################
# 1. download the sea-vl dataset using the snapshot_download below
# 2. run this script from the same folder to get the preprocessed version!
# 3. estimated time to run: 50 hours --> resume option for when you are on a cluster
# with limited running time
###################################

import numpy as np
from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
import time
import subprocess
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import re
import shutil

# Get filename and remove extension
DS_NAME = os.path.basename(__file__).split(".")[0] 
# Temporary this for download
TEMP_DIR = f"./{DS_NAME}_temp"
# Final directory for the images
IMAGES_DIR = f"./{DS_NAME}/images"
# Final path for annotations
ANNOTATIONS_FILE = f"./{DS_NAME}/preprocessed.jsonl"
# Pattern to detect and correct BBOXES
BBOXES_PATTERN = r"<bbox>(\d+,\d+,\d+,\d+)</bbox>"

class Corrector:

    def __init__(self, base_path="docstruct4m", bbox_pattern=r"<bbox>(\d+,\d+,\d+,\d+)</bbox>"):
        self.base_path = base_path
        self.width = None
        self.height = None
        self.bbox_pattern = bbox_pattern

        try:
            import imagesize
            self.get_size = lambda path: imagesize.get(f"./{DS_NAME}/{path}")
        except:
            print("Image size not found, using Pillow (3x slower).")
            from PIL import Image
            self.get_size = lambda path: Image.open(path).convert("RGB").size
        finally:
            import re


    def normalize_coordinates(self, box, image_width, image_height):
        x1, y1, x2, y2 = box
        normalized_box = [
            round((x1 / image_width) * 1000),
            round((y1 / image_height) * 1000),
            round((x2 / image_width) * 1000),
            round((y2 / image_height) * 1000)
        ]
        return normalized_box

    def store_image_size(self, image_path):
        # load_whathever
        width, height = self.get_size(image_path)
        self.width = width
        self.height = height

    def replace_bbox(self, match):
        nums = match.group(1).split(',')
        nums = [int(n) for n in nums]
        nbox = self.normalize_coordinates(nums, self.width, self.height)
        nbox = [str(n) for n in nbox]
        
        return f"<box>[[{', '.join(nbox)}]]</box>"
    
    def is_bbox(self, conversations):
        return bool(re.findall(r'<bbox>(\d+),(\d+),(\d+),(\d+)</bbox>', str(conversations)))
    
    def clean_annotation(self, image, conversations):
        # Load image to get size
        self.store_image_size(image)
        new_convs = re.sub(BBOXES_PATTERN, self.replace_bbox, str(conversations))
        
        return eval(new_convs)
    

def extract_and_remove(path, extract_dir=None):
    # Extract the directory name from the zip file path
    if extract_dir is None:
        extract_dir = "/".join(path.split("/")[:-1])

    # Unzip the file
    shutil.unpack_archive(path, extract_dir, "tar")
    print(f"Extracted to: {extract_dir}", flush=True)

    # Remove the original zip file
    os.remove(path)
    print(f"Removed original file: {path}", flush=True)

def remove_parts(directory):
    try:
        for filename in os.listdir(directory):
            if "part" in filename:
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)


def main(download=False):
    print(f"Processing {DS_NAME}", flush=True)
    
    if download:
        print("Download only mode", flush=True)
        snapshot_download(
            repo_id="mPLUG/DocStruct4M",
            repo_type="dataset",
            local_dir=TEMP_DIR,
        )
        print("Download completed!", flush=True)
        return 0

    os.makedirs(IMAGES_DIR, exist_ok=True)
    # Merge TAR 
    cat_tar_command = lambda root, tar, out: f"cat {' '.join(sorted([root+'/'+name for name in os.listdir(root) if tar in name]))} > {root+'/'+out}"
    
    print(f"Concatenating TAR", flush=True)
    print(cat_tar_command(TEMP_DIR, "partial-imgs", "train_images.tar"))
    subprocess.run(cat_tar_command(TEMP_DIR, "partial-imgs", "train_images.tar"), shell=True)
    

    print("Extracting tar", flush=True)
    extract_and_remove(f"{TEMP_DIR}/train_images.tar", IMAGES_DIR)


    # Removing previous parts
    print("Removing previous parts", flush=True)
    remove_parts(TEMP_DIR)


    # Read first json
    with open(f"{TEMP_DIR}/struct_aware_parse.jsonl", "r") as f:
        data1 = [json.loads(line) for line in f]

    with open(f"{TEMP_DIR}/multi_grained_text_localization.jsonl", "r") as f:
        data2 = [json.loads(line) for line in f]

    original_annotations = data1 + data2

    annotations = []
    count = -1

    corrector = Corrector(base_path="./docstruct4m")

    with open(ANNOTATIONS_FILE, "r") as f:
        original_annotations = [json.loads(line) for line in f]
    

    for annotation in tqdm(original_annotations, ascii=True, total=len(original_annotations), desc="Processing annotations"):
        if os.path.exists(f"{IMAGES_DIR}/{annotation['image'][0].replace('./', '')}"):
            count += 1
            prompt = annotation["messages"][0]["content"].replace("<|image|>", "")
            response = annotation["messages"][1]["content"].replace("<|image|>", "")

            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            
            if image_beginning:
                convs = [{"from":"human", "value":"<image>{}".format(prompt)}, {"from":"gpt", "value":response}]
            else:
                convs = [{"from":"human", "value":"{}<image>".format(prompt)}, {"from":"gpt", "value":response}]
            

            image_path = f"images/{annotation['image'][0].replace('./', '')}"

            # Correcting wrong bboxes (only 1/4 samples has it)
            if corrector.is_bbox(convs):
                convs = corrector.clean_annotation(image_path, convs)

            annotations.append(
                {
                    "id": count,
                    "image": image_path,
                    "conversations": convs 
                }
            )
            

    with open(ANNOTATIONS_FILE, 'w') as f:
        for entry in annotations:
            f.write(json.dumps(entry) + '\n')
    
    shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download", action="store_true")
    args = parser.parse_args()
    main(args.download)

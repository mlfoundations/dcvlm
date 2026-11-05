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

# Added after generating the preprocessed.jsonl
TOTAL_ANNOTATIONS = 7285468

# Pattern to detect and correct BBOXES
BBOXES_PATTERN = r"\[(\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+)\]"

class Corrector:

    def __init__(self, base_path="multiui", bbox_pattern=r"\[(\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+)\]"):
        self.base_path = base_path
        self.bbox_pattern = bbox_pattern
        self.exclude_pattern_str = ["(top-left x, top-left y, bottom-right x, bottom-right y)", "between 0 and 1", "floating point"]

    def replace_bbox(self, match):
        nums = match.group(1).split(', ')
        nums = [float(n) for n in nums]
        nbox = [n * 1000 for n in nums]
        nbox = [str(int(n)) for n in nbox]
        
        return f"<box>[[{', '.join(nbox)}]]</box>"
    
    def is_bbox(self, conversations):
        return bool(re.findall(self.bbox_pattern, str(conversations[-1]["value"])))
    
    def clean_annotation(self, conversations):
        
        new_convs = re.sub(self.bbox_pattern, self.replace_bbox, str(conversations))
        convs = eval(new_convs)

        # Remove reference to floating point and coordinate features
        # as in exclude pattern
        human = convs[0]["value"].split("\n")
        human_keep = []

        for single_conv in human:
            scp = single_conv.split(".")
            single_conv_keep = []
            for part in scp:
                added = False
                add = True
                for exc in self.exclude_pattern_str:
                    if exc in part:
                        add = False
                if add:
                    single_conv_keep.append(part)
            human_keep.append(".".join(single_conv_keep))

        new_human_convs = "\n".join(human_keep)        
        convs[0]["value"] = new_human_convs

        return convs
    

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
            repo_id="neulab/MultiUI",
            repo_type="dataset",
            local_dir=TEMP_DIR,
        )
        print("Download completed!", flush=True)
        return 0

    os.makedirs(IMAGES_DIR, exist_ok=True)
    # Merge TAR 
    cat_tar_command = lambda root, tar: f"cat {' '.join(sorted([root+'/'+name for name in os.listdir(root) if tar in name]))} > {root+'/'+tar}"
    
    print(f"Concatenating TAR", flush=True)
    print(cat_tar_command(TEMP_DIR, "v0.6_5M.tar"))
    subprocess.run(cat_tar_command(TEMP_DIR, "v0.6_5M.tar"), shell=True)
    print(f"V0.6 Done", flush=True)
    subprocess.run(cat_tar_command(TEMP_DIR, "v0.7_exclude_v0.6.tar"), shell=True)
    print(f"V0.7 Done", flush=True)
    subprocess.run(cat_tar_command(TEMP_DIR, "v0.8_exclude_v0.7.tar"), shell=True)
    print(f"V0.8 Done", flush=True)

    print("Extracting tar", flush=True)
    extract_and_remove(f"{TEMP_DIR}/v0.6_5M.tar", IMAGES_DIR)
    extract_and_remove(f"{TEMP_DIR}/v0.7_exclude_v0.6.tar", IMAGES_DIR)
    extract_and_remove(f"{TEMP_DIR}/v0.8_exclude_v0.7.tar", IMAGES_DIR)


    # Removing previous parts
    print("Removing previous parts", flush=True)
    remove_parts(TEMP_DIR)

    annotations = []
    count = -1

    corrector = Corrector(base_path=f"./{DS_NAME}")

    # Read first json
    with open(f"{TEMP_DIR}/stage1_data.json", "r") as f:
        original_annotations = json.load(f)

    for annotation in tqdm(original_annotations, ascii=True, total=len(original_annotations), desc="Processing Stage1"):
        if os.path.exists(f"{IMAGES_DIR}/{annotation['image']}"):
            count += 1

            if corrector.is_bbox(annotation["conversations"]):
                annotation["conversations"] = corrector.clean_annotation(annotation["conversations"])

            annotations.append(
                {
                    "id": count,
                    "image": f"images/{annotation['image']}",
                    "conversations": annotation["conversations"] 
                }
            )

    with open(ANNOTATIONS_FILE, 'w') as f:
        for entry in annotations:
            f.write(json.dumps(entry) + '\n')

    # Reset annotations but not count
    annotations = []

    with open(f"{TEMP_DIR}/stage2_data_to_be_combined_with_general_data.json", "r") as f:
        original_annotations = json.load(f)

    for annotation in tqdm(original_annotations, ascii=True, total=len(original_annotations), desc="Processing Stage2"):
        if os.path.exists(f"{IMAGES_DIR}/{annotation['image']}"):
            count += 1

            if corrector.is_bbox(annotation["conversations"]):
                annotation["conversations"] = corrector.clean_annotation(annotation["conversations"])

            annotations.append(
                {
                    "id": count,
                    "image": f"images/{annotation['image']}",
                    "conversations": annotation["conversations"] 
                }
            )

    with open(ANNOTATIONS_FILE, 'a') as f:
        for entry in annotations:
            f.write(json.dumps(entry) + '\n')
    
    shutil.rmtree(TEMP_DIR)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download", action="store_true")
    args = parser.parse_args()
    main(args.download)
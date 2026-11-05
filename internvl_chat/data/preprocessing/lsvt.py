# Takes approx 30mins

import argparse
from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
import subprocess
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import gdown

# Get filename and remove extension
DS_NAME = os.path.basename(__file__).split(".")[0] 
# Temporary this for download
TEMP_DIR = f"./{DS_NAME}_temp"
# Final directory for the images
IMAGES_DIR = f"./{DS_NAME}/images"
# Final path for annotations
ANNOTATIONS_FILE = f"./{DS_NAME}/preprocessed.jsonl"
# Prompt to be used for the caption dataset
# taken from: https://arxiv.org/pdf/2304.08485 tab. 11

OCR_PROMPTS = [
    "Please read all the characters present in the image.",
    "Perform the task of optical character recognition and output all the characters that you see in the image, in sequence.",
    "Extract and return the complete text sequence from the given image.",
    "Identify and transcribe all the characters present in the image accurately.",
    "Recognize and convert the text present in the image into machine-readable format.",
    "Read and extract the exact sequence of characters from the image without any modifications.",
    "Perform OCR on the image and provide the output as a continuous character sequence.",
    "Retrieve all visible text from the image and present it in a structured format.",
    "Analyze the given image and return the complete sequence of detected characters.",
    "Detect and extract all letters, numbers, and symbols from the image in the correct order.",
    "Carefully read and transcribe every character appearing in the image as it is.",
    "Identify, recognize, and output all text characters in the image while preserving their sequence.",
    "Extract the ground-truth text from the given image using optical character recognition.",
    "Convert the text present in the image into digital form by extracting all characters.",
    "Look at the provided image and return the sequence of text exactly as it appears."
]


def extract_and_remove(path, extract_dir=None):
    # Extract the directory name from the zip file path
    if extract_dir is None:
        extract_dir = "/".join(path.split("/")[:-1])

    # Unzip the file
    shutil.unpack_archive(path, extract_dir)
    print(f"Extracted to: {extract_dir}")

    # Remove the original zip file
    os.remove(path)
    print(f"Removed original file: {path}")


def main():

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR.split("images")[0], exist_ok=True)

    # Download and unzip dataset
    # I hope the drive link doesn't expire
    id = "18c4u7W_dDacJhe-mABYjleX3CekHb6lN"
    output = f"{TEMP_DIR}/LSVT.zip"
    gdown.download(id=id, output=output)
    extract_and_remove(output)

    # The second task use the same images as the first one having a crop image for each line
    # of text --> easier for the OCR task, otherwise we should use polygonal annotations

    for i in range(10):
        extract_and_remove(path=f"{TEMP_DIR}/train_weak_images_{i}.tar.gz", extract_dir=IMAGES_DIR)
        # This instead of mv as mv do not works with so many files (~50k) 
        # It tooks approximately 20s
        print(f"Moving images: {IMAGES_DIR}/train_weak_images_{i} --> {IMAGES_DIR}")
        subprocess.run(f"find {IMAGES_DIR}/train_weak_images_{i} -type f -exec mv " + "{}" + f" {IMAGES_DIR} " + r"\;", shell=True)
        subprocess.run(f"rm -r {IMAGES_DIR}/train_weak_images_{i}", shell=True)

    # Remove empty images
    subprocess.run(f"find {IMAGES_DIR} -type f -empty -delete", shell=True)

    try:
        # now you can safely read json
        with open(f"{TEMP_DIR}/train_weak_labels.json", "r") as f:
            annotations = json.load(f)
    except:
        # Correct error in json (oh damn)
        with open(f"{TEMP_DIR}/train_weak_labels.json", "r") as file:
            lines = file.readlines()
        
        lines[1636908] = lines[1636908].replace('1985', '"1985')

        with open(f"{TEMP_DIR}/train_weak_labels.json", "w") as file:
            file.writelines(lines)
    
        # now you can safely read json
        with open(f"{TEMP_DIR}/train_weak_labels.json", "r") as f:
            annotations = json.load(f)

    new_jsonl_entries = []

    for image_name, labels in tqdm(annotations.items(), ascii=True, total=len(annotations)):
        transcript = labels[0]["transcription"]
        image_path = os.path.join(f"images/{image_name}.jpg")
        if not os.path.exists(f"{IMAGES_DIR}/{image_name}.jpg"):
            # Raised for empty files which have been removed --> skip annotation
            continue
            
        prompt = random.choice(OCR_PROMPTS)
        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(prompt)}, {"from":"gpt", "value":transcript}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(prompt)}, {"from":"gpt", "value":transcript}]
        entry_ = {"id":len(new_jsonl_entries), "image": image_path, "conversations": convs}
        new_jsonl_entries.append(entry_)

    with open(ANNOTATIONS_FILE, 'w') as f:
        for entry in new_jsonl_entries:
            f.write(json.dumps(entry) + '\n')

    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
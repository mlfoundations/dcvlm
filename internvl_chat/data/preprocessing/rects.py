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


def unzip_and_remove(zip_path):
    # Extract the directory name from the zip file path
    extract_dir = os.path.splitext(zip_path)[0]

    # Unzip the file
    shutil.unpack_archive(zip_path, extract_dir)
    print(f"Extracted to: {extract_dir}")

    # Remove the original zip file
    os.remove(zip_path)
    print(f"Removed original file: {zip_path}")


def main():

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR.split("images")[0], exist_ok=True)

    # Download and unzip dataset
    # I hope the drive link doesn't expire
    id = "1orMtLhJt3rQl3pMoLm31eh-SmDG74W1K"
    output = f"{TEMP_DIR}/rects.zip"
    gdown.download(id=id, output=output)
    unzip_and_remove(output)

    subprocess.call(["mv", "./rects_temp/rects/img/", "./rects/"])
    subprocess.call(["mv", "./rects/img/", "./rects/images/"])

    annotations = os.listdir(f"{TEMP_DIR}/rects/gt_unicode")

    new_jsonl_entries = []

    for i, annot_name in enumerate(tqdm(annotations, ascii=True, total=len(annotations))):
        # Read the annotation file
        with open(os.path.join(f"{TEMP_DIR}/rects/gt_unicode", annot_name), "r") as f:
            annot = json.load(f)

        # Get lines in annotation
        lines = annot["lines"]
        transcript = []

        for line in lines:
            transcript.append(line["transcription"])
        
        # Since there are more lines I simply attached all of them in a single string 
        transcript = ". ".join(transcript)

        image_name = annot_name.split('.')[0]
        image_path = os.path.join(f"{IMAGES_DIR}/{image_name}.jpg")
        assert os.path.exists(image_path), f"Image {image_path} does not exist!!!"
        prompt = random.choice(OCR_PROMPTS)

        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(prompt)}, {"from":"gpt", "value":transcript}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(prompt)}, {"from":"gpt", "value":transcript}]
        entry_ = {"id":len(new_jsonl_entries), "image": f"images/{image_name}.jpg", "conversations": convs}
        new_jsonl_entries.append(entry_)

    with open(ANNOTATIONS_FILE, 'w') as f:
        for entry in new_jsonl_entries:
            f.write(json.dumps(entry) + '\n')

    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()

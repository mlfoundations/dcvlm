import os
import json
import requests
import shutil
import zipfile
import io
import gdown
import random
random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import pandas as pd
import re
from PIL import Image
import random
from tqdm import tqdm

'''
These prompts were generated using GPT-4O using the following prompt:
I am trying to create an OCR dataset for training multimodal models. I currently have an image and a ground-truth parse of all the characters present in the image. For example, this is how the ground-truth parse would look like:
'{"gt_parse": {"text_sequence": "is health had deteriorated so much that he could perf orm only nominal duties. He t ur ne d ov er the pr esid ency to Fuchs after t he season .Death and le gacy.After contractin"}}'
I want to now create a set of prompts from which I can sample for prompting the model to perform the task of OCR. For example, a simple valid prompt would be: "Please read all the characters present in the image." Another valid prompt would be: "Perform the task of optical character recognition and output all the characters that you see in the image, in sequence."
Could you please give a python list of 10-15 prompts which I can use as training prompts for constructing my OCR dataset.
'''
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


def extract_file_id(url):
    if 'id=' in url:
        return url.split('id=')[1]
    elif '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    else:
        raise ValueError("Invalid Google Drive URL")


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


download_dir = "./chinese_ocr_zip"
output_dir = "./chinese_ocr"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(download_dir, exist_ok=True)

new_jsonl_entries = []
imgs_saved = 0
cnt = 0

if not os.path.exists(os.path.join(download_dir, "ocr_dataset")):
    url = "https://drive.google.com/file/d/1pM2F04X3itsGi3n-8d7Ppxg6MFqBLKbX/view?usp=sharing"
    file_id = extract_file_id(url)
    gdown.download(f'https://drive.google.com/uc?id={file_id}', os.path.join(download_dir, "chinese_ocr.zip"),
                   quiet=False)
    extract_zip_files(download_dir)

    folder = os.listdir(download_dir)[0]
    os.rename(os.path.join(download_dir, folder), os.path.join(download_dir, "ocr_dataset"))
else:
    print("Dataset already downloaded and extracted.")

folders = [x for x in os.listdir(f"{download_dir}/ocr_dataset") if
           os.path.isdir(os.path.join(f"{download_dir}/ocr_dataset", x))]
for folder_id, folder in enumerate(folders):
    folder_name = folder.split("_")[0]
    print(f"Processing folder {folder_name}...")

    folder_path = f"{download_dir}/ocr_dataset/{folder}"
    subfolders = [x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))]

    if not subfolders:
        print(f"No subfolders in {folder}, processing images directly...")
        images = [x for x in os.listdir(folder_path) if x.endswith(('.jpg', '.jpeg', '.png'))]
        subfolder = None
    else:
        images = []
        for subfolder in subfolders:
            images.extend([(subfolder, x) for x in os.listdir(os.path.join(folder_path, subfolder)) if
                           x.endswith(('.jpg', '.jpeg', '.png'))])

    for image_id, image in tqdm(enumerate(images)):
        if subfolder is None:
            image_path = os.path.join(folder_path, image)
            subfolder_name = ""
        else:
            subfolder_name, image = image
            image_path = os.path.join(folder_path, subfolder_name, image)

        assert os.path.exists(image_path), f"Image {image_path} does not exist!"
        image = Image.open(image_path)
        width, height = image.size
        assert width is not None, 'no images actually present for sample in folder {}, sample {}'.format(folder_id,
                                                                                                         image_id)
        assert height is not None, 'no images actually present for sample in folder {}, sample {}'.format(folder_id,
                                                                                                          image_id)

        if subfolder_name == "":
            file_name = f"{output_dir}/images/{folder_name}_{image_id}.png"
        else:
            file_name = f"{output_dir}/images/{folder_name}_{subfolder_name}_{image_id}.png"
        image.save(file_name)
        imgs_saved += 1

        json_file = image_path.replace("." + image_path.split(".")[-1], ".json")
        assert os.path.exists(json_file), f"JSON file {json_file} does not exist!"

        with open(json_file, "r") as f:
            data = json.load(f)

        prompt_ = random.choice(OCR_PROMPTS)
        response_ = ", ".join(
            ann["transferText"] for ann in data["answer"]["markData"]["annotations"] if "transferText" in ann)

        # Add entry to new JSONL list
        image_beginning = random.choice([0, 1])
        if image_beginning:
            convs = [{"from": "human", "value": "<image>\n{}".format(prompt_)}, {"from": "gpt", "value": response_}]
        else:
            convs = [{"from": "human", "value": "{}\n<image>".format(prompt_)}, {"from": "gpt", "value": response_}]

        entry_ = {"id": cnt, "image": 'images/' + os.path.basename(file_name), "height": height, "width": width,
                  "conversations": convs}
        new_jsonl_entries.append(entry_)
        cnt += 1

with open(f'{output_dir}/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(download_dir)

print(
    "Preprocessing complete! Saved as '{}/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(
        output_dir, imgs_saved, output_dir, cnt))
import os
import json
import requests
import shutil
import random

random.seed(0)
import tarfile
import io
import gdown
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


def extract_tar_files(download_dir):
    """
    This is done to extract all tar files (merged split tars and regular tars) in the directory.
    """
    tar_files = [f for f in os.listdir(download_dir) if f.endswith(('.tar', '.tar.gz'))]

    for tar_file in tar_files:
        tar_path = os.path.join(download_dir, tar_file)
        print(f"Extracting {tar_file}...")

        try:
            with tarfile.open(tar_path, 'r:*') as tar_ref:
                tar_ref.extractall(download_dir)
            print(f"Extracted {tar_file}")
        except tarfile.ReadError:
            print(f"Failed to extract {tar_file}. It may be corrupted.")

        os.remove(tar_path)


download_dir = "./eaten_tars"
output_dir = "./eaten"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(download_dir, exist_ok=True)

new_jsonl_entries = []
imgs_saved = 0
cnt = 0
bc_cnt = 0
pp_cnt = 0
tt_cnt = 0

# make a list of folders and check if they all exist
folders = ['dataset_business', 'dataset_passport', 'dataset_trainticket']

if all(os.path.exists(os.path.join(download_dir, folder)) for folder in folders):
    print("All required folders exist. No download needed.")
else:
    url = "https://drive.google.com/file/d/10QPeNGbsCzUOdxESuiA6CROHVx2AkITb/view?usp=sharing"
    file_id = extract_file_id(url)
    gdown.download(f'https://drive.google.com/uc?id={file_id}', os.path.join(download_dir, "eaten.tar"), quiet=False)
    extract_tar_files(download_dir)

# Start with business
folder = 'dataset_business'
df_business = pd.read_pickle(f"{download_dir}/{folder}/business.pkl")

for filename, data in tqdm(df_business.items(), desc="Processing business cards", leave=True, dynamic_ncols=True):
    values = list(data.values())
    comma_separated = ", ".join(values)

    match = re.search(r'(\d+)', filename)
    if match:
        image_number = int(match.group(1))

        folder_bases = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000,
                        90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000]

        folder_base = max([base for base in folder_bases if base <= image_number])
        folder_name = f"images_{folder_base}"

        image_name = folder_name
    else:
        continue
    image_path = f"{download_dir}/{folder}/train/{folder_name}/{filename}"
    if not os.path.exists(
            image_path):  # there are samples which are referenced in the annotations but are not there in the images
        # print(f"Image {image_path} exists in annotations but not in images")
        continue
    assert os.path.exists(image_path), f"Image {image_path} does not exist"

    image = Image.open(image_path)
    output_image_path = f"{output_dir}/images/bc_{bc_cnt}.png"
    shutil.copy(image_path, output_image_path)
    imgs_saved += 1

    width, height = image.size
    assert width is not None, 'no images actually present for sample in business_cards, sample {}'.format(filename)
    assert height is not None, 'no images actually present for sample in business_cards, sample {}'.format(filename)
    assert output_image_path is not None, 'no images actually present for sample in business_cards, sample {}'.format(filename)

    # add entry to new jsonl list
    image_beginning = random.choice([0, 1])
    prompt_ = random.choice(OCR_PROMPTS)
    if image_beginning:
        convs = [{"from": "human", "value": "<image>\n{}".format(prompt_)}, {"from": "gpt", "value": comma_separated}]
    else:
        convs = [{"from": "human", "value": "{}\n<image>".format(prompt_)}, {"from": "gpt", "value": comma_separated}]

    entry_ = {"id": cnt, "image": 'images/' + os.path.basename(output_image_path), "height": height, "width": width,
              "conversations": convs}
    new_jsonl_entries.append(entry_)
    cnt += 1
    bc_cnt += 1

# Then passport
folder = 'dataset_passport'
df_passport = pd.read_pickle(f"{download_dir}/{folder}/10w_line_text_synth.pkl")
for filename, data in tqdm(df_passport.items(), desc="Processing passports", leave=True, dynamic_ncols=True):
    values = list(data.values())
    comma_separated = ", ".join(values)
    image_path = f"{download_dir}/{folder}/train/{filename}.jpg"
    if not os.path.exists(
            image_path):  # there are samples which are referenced in the annotations but are not there in the images
        # print(f"Image {image_path} exists in annotations but not in images")
        continue
    assert os.path.exists(image_path), f"Image {image_path} does not exist"

    image = Image.open(image_path)
    output_image_path = f"{output_dir}/images/pp_{pp_cnt}.png"
    shutil.copy(image_path, output_image_path)
    imgs_saved += 1

    width, height = image.size
    assert width is not None, 'no images actually present for sample in passport dataset, sample {}'.format(filename)
    assert height is not None, 'no images actually present for sample in passport dataset, sample {}'.format(filename)
    assert output_image_path is not None, 'no images actually present for sample in passport dataset, sample {}'.format(filename)

    # add entry to new jsonl list
    image_beginning = random.choice([0, 1])
    prompt_ = random.choice(OCR_PROMPTS)
    if image_beginning:
        convs = [{"from": "human", "value": "<image>\n{}".format(prompt_)}, {"from": "gpt", "value": comma_separated}]
    else:
        convs = [{"from": "human", "value": "{}\n<image>".format(prompt_)}, {"from": "gpt", "value": comma_separated}]

    entry_ = {"id": cnt, "image": 'images/' + os.path.basename(output_image_path), "height": height, "width": width,
              "conversations": convs}
    new_jsonl_entries.append(entry_)
    cnt += 1
    pp_cnt += 1

# Then train ticket
folder = 'dataset_trainticket'
# df_real = pd.read_pickle(f"{download_dir}/{folder}/real_1920.pkl") #can't use because no filepath mapping, just filenames
df_synthetic = pd.read_pickle(f"{download_dir}/{folder}/synth_300k.pkl")

for filename, data in tqdm(df_synthetic.items(), desc="Processing train tickets", leave=True, dynamic_ncols=True):
    values = list(data.values())
    comma_separated = ", ".join(values)
    image_path = f"{download_dir}/{folder}/train/{filename}.jpg"
    if not os.path.exists(
            image_path):  # there are samples which are referenced in the annotations but are not there in the images
        # print(f"Image {image_path} exists in annotations but not in images")
        continue
    assert os.path.exists(image_path), f"Image {image_path} does not exist"

    image = Image.open(image_path)
    output_image_path = f"{output_dir}/images/tt_{tt_cnt}.png"
    shutil.copy(image_path, output_image_path)
    imgs_saved += 1

    width, height = image.size
    assert width is not None, 'no images actually present for sample in train ticket dataset, sample {}'.format(filename)
    assert height is not None, 'no images actually present for sample in train ticket dataset, sample {}'.format(filename)
    assert output_image_path is not None, 'no images actually present for sample in train ticket dataset, sample {}'.format(filename)

    # add entry to new jsonl list
    image_beginning = random.choice([0, 1])
    prompt_ = random.choice(OCR_PROMPTS)
    if image_beginning:
        convs = [{"from": "human", "value": "<image>\n{}".format(prompt_)}, {"from": "gpt", "value": comma_separated}]
    else:
        convs = [{"from": "human", "value": "{}\n<image>".format(prompt_)}, {"from": "gpt", "value": comma_separated}]

    entry_ = {"id": cnt, "image": 'images/' + os.path.basename(output_image_path), "height": height, "width": width,
              "conversations": convs}
    new_jsonl_entries.append(entry_)
    cnt += 1
    tt_cnt += 1

with open(f'{output_dir}/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(download_dir)

print(
    "Preprocessing complete! Saved as '{}/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(
        output_dir, imgs_saved, output_dir, cnt))
"""
SROIE 2019 text recognition dataset
https://huggingface.co/datasets/priyank-m/SROIE_2019_text_recognition
"""

import os
import random
from huggingface_hub import snapshot_download
from tqdm import tqdm
import json
import shutil
import zipfile
from tqdm import tqdm
import pandas as pd

random.seed(0)

OUTPUT_DIR = "./sroie_2019"

os.makedirs(OUTPUT_DIR, exist_ok=True)
snapshot_download(
    repo_id="priyank-m/SROIE_2019_text_recognition",
    repo_type="dataset",
    allow_patterns=["train.zip"],  # Only download files in the data folder
    local_dir=f"{OUTPUT_DIR}/tmp",
)

print("Extracting images")
with zipfile.ZipFile(f"{OUTPUT_DIR}/tmp/train.zip") as images_zip:
    images_zip.extractall(f"{OUTPUT_DIR}")
shutil.move(f"{OUTPUT_DIR}/train", f"{OUTPUT_DIR}/images")
with open(f"{OUTPUT_DIR}/images/metadata.jsonl", "r") as f:
    metadata = f.readlines()
    data = [json.loads(line) for line in metadata]

PROMPTS = [
    "Extract and transcribe all visible text from this image.",
    "Identify and transcribe any readable text present in the image.",
    "Please recognize and extract all text appearing in the image.",
    "Analyze this image and return the transcribed text exactly as it appears.",
    "Read the text in this image and provide an accurate transcription.",
    "Detect and transcribe all characters, words, and sentences found in the image.",
    "Convert any textual content in this image into a machine-readable format.",
    "Perform OCR on this image and return the extracted text.",
    "Scan the image for text and provide the full transcription.",
    "Retrieve and format all text content present in the image.",
    "Identify and return the text captured in this image while preserving formatting.",
    "Extract text from the image, keeping line breaks and spacing intact.",
    "Transcribe any text from the image, including handwritten or printed words.",
    "Analyze the image for text and return the extracted content in structured format.",
    "Perform text recognition on this image and return an accurate transcription.",
    "Read and extract all visible words from this image as accurately as possible.",
    "Identify all alphanumeric characters in the image and return them as text.",
    "Transcribe the textual content of this image while maintaining its original structure.",
    "Perform a thorough text extraction from this image and return the results.",
    "Detect and transcribe text from the image, handling different fonts and sizes.",
    "Extract all embedded text from this image, even if partially obscured.",
    "Recognize and return the textual information present in this image in plain text format.",
    "Read and output the words from this image with high accuracy."
]
"""
These prompts a generated using GPT-4o with this prompt:
> I have images and their transcribed text. Create a list of prompts I can use for an LLM to transcribe the text in an image.
> The format should be PROMPTS = [...]
"""

preprocessed, cnt = [], 0
for entry in tqdm(data, desc=f"Processing rows", total=len(data)):
    image_path = os.path.join("images", entry["file_name"])
    query = random.choice(PROMPTS)
    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    value_str = f"<image>{query}" if image_beginning else f"{query}<image>"
    convs = [
        {"from": "human", "value": value_str},
        {"from": "gpt", "value": entry["text"]},
    ]
    assert os.path.exists(
        os.path.join(OUTPUT_DIR, image_path)
    ), "Image in dataset does not exist on filesystem."
    preprocessed.append({
        "id": cnt,
        "image": image_path,
        "conversations": convs,
    })
    cnt += 1

shutil.rmtree(f"{OUTPUT_DIR}/tmp")
os.remove(f"{OUTPUT_DIR}/images/metadata.jsonl")

with open(f"{OUTPUT_DIR}/preprocessed.jsonl", "w") as f:
    for entry in tqdm(preprocessed, desc="Writing to jsonl", total=len(preprocessed)):
        f.write(json.dumps(entry) + "\n")

print(f"Downloaded SROIE 2019 dataset to {OUTPUT_DIR}")

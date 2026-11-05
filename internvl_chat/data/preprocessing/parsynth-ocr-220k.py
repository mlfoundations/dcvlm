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
import tarfile

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

download_dir = "./parsynth_ocr_200k_parquets"
output_dir = "./parsynth_ocr_200k"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

snapshot_download(
    repo_id="hezarai/parsynth-ocr-200k",
    repo_type="dataset",
    allow_patterns=["data/train-*"],
    local_dir=download_dir)

new_jsonl_entries = []
cnt = 0
imgs_saved = 0

parquets = [x for x in os.listdir(f"{download_dir}/data") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):

    df = pd.read_parquet(os.path.join(f"{download_dir}/data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):

        image_bytes = row["image_path"]["bytes"]
        text = row["text"]

        # assert image is not None, 'image is None!!!' and text is a string
        assert image_bytes is not None, 'image bytes are None!!!'
        assert text is not None, 'text is None!!!'

        image = Image.open(io.BytesIO(image_bytes))

        file_name = f"{output_dir}/images/sample_{cnt}.png"
        image.save(file_name)
        imgs_saved += 1

        width, height = image.size
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index,
                                                                                                          sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(
            parquet_index, sample_index)
        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(
            parquet_index, sample_index)

        # add entry to new jsonl list
        image_beginning = random.choice([0, 1])
        prompt_ = random.choice(OCR_PROMPTS)
        if image_beginning:
            convs = [{"from": "human", "value": "<image>\n{}".format(prompt_)}, {"from": "gpt", "value": text}]
        else:
            convs = [{"from": "human", "value": "{}\n<image>".format(prompt_)}, {"from": "gpt", "value": text}]
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



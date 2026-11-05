"""
FigureQA dataset
https://www.microsoft.com/en-us/research/project/figureqa-dataset/downloads/
"""

import os
import requests
import random
from tqdm import tqdm
import json
import shutil
import tarfile
from tqdm import tqdm

random.seed(0)

OUTPUT_DIR = "./figureqa"

os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)

def download_and_extract(url, output_dir):
    local_filename = os.path.join(output_dir, url.split("/")[-1])
    if not os.path.exists(local_filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    with tarfile.open(local_filename, "r:gz") as f:
        f.extractall(output_dir)

print("Downloading FigureQA dataset")
download_and_extract('https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508/figureqa-train1-v1.tar.gz', OUTPUT_DIR)
shutil.move(f'{OUTPUT_DIR}/train1/png', f'{OUTPUT_DIR}/images')

with open(f'{OUTPUT_DIR}/train1/qa_pairs.json', 'r') as f:
    data = json.load(f)
data = data['qa_pairs']

preprocessed = []
for i, entry in tqdm(enumerate(data), desc='Processing entries', total=len(data)):
    image_path = os.path.join('images', f'{entry["image_index"]}.png')
    assert os.path.exists(os.path.join(OUTPUT_DIR, image_path)), 'Image in dataset does not exist on filesystem.'
    image_beginning = random.choice([0, 1])
    question = entry['question_string']
    value_str = f'<image>{question}' if image_beginning else f'{question}<image>'
    convs = [
        {'from': 'human', 'value': value_str},
        {'from': 'gpt', 'value': "Yes" if entry["answer"] else "No"},
    ]
    preprocessed.append({
        'id': i,
        'image': image_path,
        'conversations': convs
    })

with open(f"{OUTPUT_DIR}/preprocessed.jsonl", "w") as f:
    for entry in tqdm(preprocessed, desc="Writing to jsonl", total=len(preprocessed)):
        f.write(json.dumps(entry) + "\n")

os.remove(f'{OUTPUT_DIR}/figureqa-train1-v1.tar.gz')
shutil.rmtree(f'{OUTPUT_DIR}/train1')

print(f"Downloaded FigureQA dataset to {OUTPUT_DIR}")

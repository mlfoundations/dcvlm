from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

output_dir = './lvis_instruct_4v'
os.makedirs(output_dir, exist_ok=True)

ANNOTS_URL = 'https://huggingface.co/datasets/X2FD/LVIS-Instruct4V/resolve/main/lvis_instruct4v_220k.json'
assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

snapshot_download(
    repo_id="X2FD/LVIS-Instruct4V",
    repo_type="dataset",
    allow_patterns=["*.json"],  # Only download json file
    local_dir="lvis_instruct_4v"
)

with open(os.path.join(output_dir, 'lvis_instruct4v_220k.json')) as f:
    json_contents = json.load(f)

preprocessed = []

for index, json_contents in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_path = json_contents['image']
    convs = json_contents['conversations']

    assert convs is not None, 'conversations is None!'
    assert image_path is not None, 'image_path is None!'

    _, split, img_p = image_path.split('/')

    processed_path = 'images/{}/{}/{}'.format(split, split, img_p)
    assert os.path.exists(os.path.join(os.environ['COCO_DIR'], processed_path)), 'Image {} does not exist!!'.format(processed_path)

    entry_ = {"id":len(preprocessed), "image": processed_path, "conversations": convs}
    preprocessed.append(entry_)

with open('./lvis_instruct_4v/preprocessed.jsonl', 'w') as f:
    for entry in preprocessed:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'lvis_instruct_4v/preprocessed.jsonl' with {} total samples.".format(len(preprocessed)))

os.remove(os.path.join(output_dir, 'lvis_instruct4v_220k.json'))

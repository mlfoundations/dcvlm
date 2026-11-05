from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import re

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

patterns = ["rec_conversation_22k.json", "rec_detailed_description_42k.json", "rec_region_captioning_63k.json"]

snapshot_download(
    repo_id="OpenGVLab/AS-V2",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir="as_data"
)

output_dir = "./as_v2"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./as_data") if x.endswith(".json")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./as_data", jsonl_file))
    for sample_index, row in tqdm(df.iterrows(), ascii=True, total=len(df)):
        # Change image path to common COCO format
        image = row["image"].split("/")[-1]
        image = os.path.join("images/train2017/train2017", image)

        assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image)), 'Image {} does not exist!!!'.format(image)

        height = row["height"]
        width = row["width"]
        conversations = row["conversations"]

        entry_ = {"id":n_samples, "image":image, "height":height, "width":width, "conversations":row["conversations"]}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./as_v2/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('as_data')

print("Preprocessing complete! Saved as 'as_v2/preprocessed.jsonl' with {} total samples.".format(n_samples))
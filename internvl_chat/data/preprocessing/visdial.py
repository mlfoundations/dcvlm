from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

snapshot_download(
    repo_id="HuggingFaceM4/VisDial",
    repo_type="dataset",
    allow_patterns=["data/train*parquet"],
    local_dir="visdial_parquets"
)

output_dir = "./visdial"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
parquets = [x for x in os.listdir("./visdial_parquets/data") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./visdial_parquets/data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image = row["image_path"]
        dialog = row["dialog"]

        # ensure image exists
        assert image is not None, 'image path is None!!!'
        if 'train2014' in os.path.basename(image):
            assert os.path.exists(os.path.join(os.environ['COCO_DIR'], 'images/train2014/train2014', os.path.basename(image))), 'image path {} is not available!!!'.format(image)
        elif 'val2014' in os.path.basename(image):
            assert os.path.exists(os.path.join(os.environ['COCO_DIR'], 'images/val2014/val2014', os.path.basename(image))), 'image path {} is not available!!!'.format(image)
        else:
            raise ValueError("image is not from train or val set!!!")

        assert dialog is not None, 'dialog is None!!!'

        # add entry to new jsonl list
        convs = []
        for index, t in enumerate(dialog):
            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if index == 0:
                if image_beginning:
                    human_text = '<image>{}'.format(t[0])
                else:
                    human_text = '{}<image>'.format(t[0])
            else:
                human_text = '{}'.format(t[0])

            convs.append({"from":"human", "value":human_text})
            convs.append({"from":"gpt", "value":t[1]})

        if 'train2014' in os.path.basename(image):
            im_p = os.path.join('images/train2014/train2014', os.path.basename(image))
        else:
            im_p = os.path.join('images/val2014/val2014', os.path.basename(image))
        entry_ = {"id":len(new_jsonl_entries), "image": im_p, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open('./visdial/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('visdial_parquets')

print("Preprocessing complete! Saved as 'visdial/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
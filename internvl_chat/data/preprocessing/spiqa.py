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
import zipfile

download_dir = "./spiqa_files"
output_dir = "./spiqa"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

snapshot_download(
    repo_id="google/spiqa",
    repo_type="dataset",
    allow_patterns=["train_val/SPIQA_train.json","train_val/SPIQA_train_val_Images.zip"],
    local_dir=download_dir)

new_jsonl_entries = []
cnt = 0
imgs_saved = 0

if os.path.exists(os.path.join(download_dir, "SPIQA_train_val_Images")):
    print("Images already unzipped")
else:
    zip_file = os.path.join(download_dir,"train_val/SPIQA_train_val_Images.zip")
    print(f"Unzipping {zip_file}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    print("Unzipped images")

#load the json file
with open(os.path.join(download_dir, "train_val/SPIQA_train.json"), 'r') as f:
    data = json.load(f)
print("Read json file with {} entries".format(len(data)))

for idx, row in enumerate(tqdm(data, ascii=True)):
    paper_id = data[row]["paper_id"]
    for qa_pair in data[row]['qa']:
        prompt_ = qa_pair['question']
        response_ = qa_pair['answer']
        image = qa_pair['reference']
        file_path = os.path.join(download_dir, "SPIQA_train_val_Images", paper_id, image)
        assert os.path.exists(file_path), 'Image {} does not exist!!!'.format(file_path)

        file_name = f"{output_dir}/images/{image}"
        shutil.copy(file_path, file_name)
        imgs_saved += 1

        image = Image.open(file_path)
        width, height = image.size
        assert width is not None, 'no images actually present for sample {}'.format(idx)
        assert height is not None, 'no images actually present for sample {}'.format(idx)
        assert file_name is not None, 'no images actually present for sample {}'.format(idx)

        # add entry to new jsonl list
        image_beginning = random.choice([0, 1])
        cnt += 1
        if image_beginning:
            convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":response_}]
        else:
            convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
        entry_ = {"id":cnt, "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(f'{output_dir}/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(download_dir)

print("Preprocessing complete! Saved as '{}/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(output_dir, imgs_saved, output_dir, cnt))
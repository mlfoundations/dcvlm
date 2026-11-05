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

download_dir = "./chartgemma_parquets"
output_dir = "./chartgemma"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

snapshot_download(
    repo_id="ahmed-masry/ChartGemma",
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

        image_bytes = row['image']
        prompt_ = row["input"]
        response_ = row["output"]

        assert image_bytes is not None, 'image bytes are None!!!'
        assert prompt_ is not None, 'prompt is None!!!'
        assert response_ is not None, 'response is None!!!'

        image = Image.open(io.BytesIO(image_bytes))
        file_name = f"{output_dir}/images/parquet_{parquet_index}_sample_{sample_index}.png"
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

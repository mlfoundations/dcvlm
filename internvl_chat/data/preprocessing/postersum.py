##### NOTE: this dataset has been uploaded on the internal drive since it might take a while to run the full script to download.
##### Hence, if you are feeling particularly impatient, you can just download the full data folder (that contains both images and preprocessed jsonl) instead of running this script.

import json
from huggingface_hub import snapshot_download
import os
import pandas as pd
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import shutil
import requests

# taken from https://arxiv.org/pdf/2502.17540 appendix B
PROMPT = "Write an abstract for an AI conference paper for the given research poster image."

######### Utility files

def download_image(url, output_dir, filename):
    """Download an image from a given URL and save it to the specified folder with a specific filename."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    else:
        print(f"Failed to download {url} - Status Code: {response.status_code}")

if 'HF_TOKEN' not in os.environ:
    raise ValueError('Please set your HF_TOKEN environment variable. If you dont have one, create one at: https://huggingface.co/settings/tokens and do export HF_TOKEN=hf_XXXXXX.')

output_dir = "./postersum"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

snapshot_download(
    repo_id="rohitsaxena/PosterSum",
    repo_type="dataset",
    allow_patterns=["data/train*.parquet"],  # Only download train file
    local_dir="postersum_parquets",
    token=os.environ['HF_TOKEN']
)

new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir("./postersum_parquets/data") if x.endswith(".parquet")]
assert len(parquets) == 1, 'More than 1 parquet available!!!'

for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./postersum_parquets/data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True, total=len(df)):
        img_url = row["image_url"]
        response_ = row["abstract"]

        assert img_url is not None, "image url is None!!!"
        assert response_ is not None, "response is None!!!"

        # save image
        download_image(img_url, output_dir+'/images', 'image_{}.png'.format(len(new_jsonl_entries)))
        imgs_saved += 1

        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(PROMPT)}, {"from":"gpt", "value":response_}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(PROMPT)}, {"from":"gpt", "value":response_}]
        entry_ = {"id":len(new_jsonl_entries), "image": 'images/'+'image_{}.png'.format(len(new_jsonl_entries)), "conversations": convs}
        new_jsonl_entries.append(entry_)

with open('./postersum/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('postersum_parquets')

print("Preprocessing complete! Saved as 'postersum/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))
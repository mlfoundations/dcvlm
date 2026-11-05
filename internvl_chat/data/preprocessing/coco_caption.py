import json
import os
import sys
import requests
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl

# taken from: https://arxiv.org/pdf/2304.08485 tab. 11
LLAVA_DESC_SHORT_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo’s key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'
KARPATHY_TRAIN_URL = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json"
KARPATHY_TRAIN_FILE = "coco_karpathy_train.json"
ROOT = './coco_caption'

######## All utility functions

# Function to get coco images path
def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2014", f"{image_id:012}")

def download_file(url, output_file):
    """Downloads a file from the given URL with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_file, "wb") as file, tqdm(
        desc=output_file,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

    print(f"Download complete: {output_file}")

######## first download the Karpathy split train json annotations 

# check if the original jsonl file already exists to avoid re-downloading
if not os.path.exists(KARPATHY_TRAIN_FILE):
    download_file(KARPATHY_TRAIN_URL, KARPATHY_TRAIN_FILE)
else:
    print(f"File '{KARPATHY_TRAIN_FILE}' already exists. Skipping download.")

# create dataset directory if it does not exist
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

######## We now want to start the reformatting procedure

# load the Karpathy train split json annotations
coco_karpathy_train_annots = json.load(open(KARPATHY_TRAIN_FILE, 'r'))

new_jsonl_entries = []

for index, instance in tqdm(enumerate(coco_karpathy_train_annots), ascii=True, total=len(coco_karpathy_train_annots)):

    if 'val2014' in instance['image']:
        image_path = get_coco_path('val', instance['image'], 'images')
    elif 'train2014' in instance['image']:
        image_path = get_coco_path('train', instance['image'], 'images')
    else:
        raise Exception("Data must come from one of ['val2014', 'train2014']")

    assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image_path)), 'Image {} does not exist!!!'.format(image_path)
    
    # select a prompt from one of the LLaVA short caption descriptions
    prompt_ = random.choice(LLAVA_DESC_SHORT_PROMPTS)

    # response should simply be the ground truth caption
    response_ = instance['caption']
    
    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    if image_beginning:
        convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    
    entry_ = {"id": len(new_jsonl_entries), "image": image_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./coco_caption/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove(KARPATHY_TRAIN_FILE)
print("Preprocessing complete! Saved as 'coco_caption/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

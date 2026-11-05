import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import requests

'''
Sourced from gpt-4O from the chatgpt-UI with the following prompt:

I have a dataset of images of common objects in context (COCO). I have corresponding caption descriptions of the spatial relations between the two objects present in each image. For each of these images, the task is to determine whether the corresponding caption belongs to the given image or not, through binary classification. I want to convert this dataset into an instruction tuning dataset where the prompt would be something like "Is this statement about the image True or False?". Please provide a list of 15 different such prompts I can use as part of the visual instruction tuning dataset that I can randomly sample when constructing the dataset.
'''

PROMPTS_LIST = [
        "Does this statement accurately describe the image?",
        "Is the given caption a true description of the image?",
        "Determine whether the provided statement correctly represents the image.",
        "Is this caption relevant to the image?",
        "Does this statement correctly describe the spatial relationship in the image?",
        "Assess whether this caption aligns with the visual content of the image.",
        "Is this description factually accurate based on the image?",
        "Evaluate if the statement provides a valid interpretation of the image.",
        "Is the caption a correct explanation of what is happening in the image?",
        "Does the given description match the spatial arrangement of objects in the image?",
        "Decide whether this statement truthfully reflects the scene depicted in the image.",
        "Would you say this caption is a correct description of the image?",
        "Can this statement be considered an accurate summary of the image content?",
        "Judge if the caption corresponds correctly to the objects and their relations in the image.",
        "Is the spatial relationship described in the caption true for this image?"
        ]

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

VSR_URL = "https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/refs/heads/master/data/splits/random/train.jsonl"
VSR_JSONL_FILE = "train.jsonl"
ROOT = './vsr'

######## All utility functions

# Function to get coco images path
def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}")

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

######## first download the existing jsonl annotations of VSR random-split train set from https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/refs/heads/master/data/splits/random/train.jsonl

# check if the original jsonl file already exists to avoid re-downloading
if not os.path.exists(VSR_JSONL_FILE):
    download_file(VSR_URL, VSR_JSONL_FILE)
else:
    print(f"File '{VSR_JSONL_FILE}' already exists. Skipping download.")

# create dataset directory if it does not exist
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

######## We now want to start the reformatting procedure

# load the original VSR random-split train set jsonl file
data_json = []
with open('train.jsonl', "r") as f:
    lines = f.readlines()
    for line in lines:
        j_line = json.loads(line)
        data_json.append(j_line)

# collect the images from VSR train set belonging to COCO train2017 dataset
train_data_json = []
for instance in data_json:
    # do not include images from the COCO val2017 split
    if 'val2017' in instance['image_link']:
        continue
    train_data_json.append(instance)


new_jsonl_entries = []

for index, instance in tqdm(enumerate(train_data_json), ascii=True, total=len(train_data_json)):
    
    # select a pre-prompt from the chatgpt generated prompts
    pre_prompt = random.choice(PROMPTS_LIST)
    gt_caption = instance['caption']
    prompt_ = "{} {}".format(pre_prompt, gt_caption)
    answer_ = "True" if instance['label'] else "False"

    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    if image_beginning: 
        convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    else:
        convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":answer_}]
    
    image_path = get_coco_path('train', instance['image'], 'images/train2017')
    assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image_path)), 'Image {} does not exist!!!'.format(image_path)

    entry_ = {"id": len(new_jsonl_entries), "image": image_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./vsr/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove('train.jsonl')
print("Preprocessing complete! Saved as 'vsr/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

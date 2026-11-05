import kagglehub
import os
import shutil
import json
from tqdm import tqdm
import csv
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl

# Set Kaggle credentials at runtime
assert 'KAGGLE_USERNAME' in os.environ, 'Please set-up your KAGGLE_USERNAME env variable for your kaggle username, set is using export KAGGLE_USERNAME=XXXXX'
assert 'KAGGLE_KEY' in os.environ, 'Please set-up your KAGGLE_KEY env variable for your kaggle key, set is using export KAGGLE_KEY=XXXXX'

ROOT_DIR = './tolokavqa'
IMAGE_DIR = ROOT_DIR + '/images'
TRAIN_ANNOTATIONS = 'train.csv'
os.makedirs(ROOT_DIR, exist_ok=True)


# Function to normalize bbox coordinates, taken from InternVL documentations
def normalize_coordinates(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    normalized_bbox = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_bbox


# Need to set-up the cache directory to prevent Kaggle from downloading it to ~/.cache
os.environ["KAGGLEHUB_CACHE"] = ROOT_DIR

# Download latest version
kaggle_path = kagglehub.dataset_download("dustalov/toloka-wsdm-cup-2023-vqa")

# Images are saved under: datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/img
print(f"Dataset saved in {kaggle_path} by Kaggle...")

# Copy images from the Kaggle-saved directory to the IMAGE_DIR
print(f"Copying the images from Kaggle-saved directory to {IMAGE_DIR}...")
shutil.move(os.path.join(kaggle_path, 'images/'), IMAGE_DIR)

# Load the original train annotations in the csv format
with open(os.path.join(kaggle_path, TRAIN_ANNOTATIONS), 'r') as f:
    train_csv_annotations = [line for line in csv.reader(f)]

# Starting to form the jsonl annotations...
new_jsonl_entries = []

for index, instance in tqdm(enumerate(train_csv_annotations[1:]), ascii=True, total=len(train_csv_annotations[1:])):

    # train_csv_annotations[0] -> ['image', 'width', 'height', 'left', 'top', 'right', 'bottom', 'question']
    image_path = instance[0].split('/')[-1]
    assert os.path.exists(os.path.join(IMAGE_DIR, image_path)), 'Image {} does not exist!!!'.format(image_path)
    
    prompt_ = instance[-1]
    
    # InternVL follows bottom-left to top-right coordinates for bbox representations
    image_width, image_height = int(instance[1]), int(instance[2])
    
    # x1y1 x2y2 -> left, top, right, bottom
    # Note: TolokaVQA annotations refer to top as bottom and bottom as top, i.e ∀i, top_i < bottom_i
    bbox_coordinates = [int(instance[3]), int(instance[4]), int(instance[5]), int(instance[6])]
    reformatted_bbox = normalize_coordinates(bbox_coordinates, image_width, image_height)
    response_ = "<ref>{}</ref><box>[{}]</box>".format(prompt_, reformatted_bbox)

     # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    if image_beginning:
        convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":response_}]

    entry_ = {"id": len(new_jsonl_entries), "image": 'images/'+ image_path, "conversations": convs, "width": image_width, "height": image_height}
    new_jsonl_entries.append(entry_)

with open(os.path.join(ROOT_DIR, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing for TolokaVQA complete! Saved as 'tolokavqa/preprocessed.jsonl' with {} total samples".format(len(new_jsonl_entries)))
shutil.rmtree(os.path.join(ROOT_DIR, 'datasets'))

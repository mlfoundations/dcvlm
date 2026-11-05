import json
import os
from PIL import Image
import random
import string
from tqdm import tqdm
import wget
import zipfile


DATASET_NAME = "visual7w"
print("Preparing dataset:", DATASET_NAME)

os.makedirs(os.path.join(DATASET_NAME, "images"), exist_ok=True)

IMAGE_URL = "http://vision.stanford.edu/yukezhu/visual7w_images.zip"
IMAGE_ZIP = "visual7w_images.zip"
TELLING_URL = "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"
TELLING_ZIP = "dataset_v7w_telling.zip"
TELLING_FILE = "dataset_v7w_telling.json"
POINTING_URL = "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_pointing.zip"
POINTING_ZIP = "dataset_v7w_pointing.zip"
POINTING_FILE = "dataset_v7w_pointing.json"

print(f"Downloading images to '{IMAGE_ZIP}'")
wget.download(IMAGE_URL, out=IMAGE_ZIP)
print(f"Extracting images to '{DATASET_NAME}'")
with zipfile.ZipFile(IMAGE_ZIP) as zfp:
    zfp.extractall(DATASET_NAME)
print("Downloading annotations")
if not os.path.exists(TELLING_FILE):
    wget.download(TELLING_URL, out=TELLING_ZIP)
    with zipfile.ZipFile(TELLING_ZIP) as zfp:
        zfp.extractall("./")
if not os.path.exists(POINTING_FILE):
    wget.download(POINTING_URL, out=POINTING_ZIP)
    with zipfile.ZipFile(POINTING_ZIP) as zfp:
        zfp.extractall("./")


# Preprocessing

print("Starting preprocessing")
random.seed(0)

out_data = []

# Telling (normal QA)
with open(TELLING_FILE) as fp:
    telling_data = json.load(fp)
for image_data in tqdm(telling_data['images'], desc="Telling QA"):
    if image_data['split'] != "train":
        continue

    image_conversations = []
    for qa_pair in image_data['qa_pairs']:
        prompt = f"Question: {qa_pair['question']}\nOptions:\n"
        choices = qa_pair['multiple_choices'] + [qa_pair['answer']]
        random.shuffle(choices)
        correct_index = choices.index(qa_pair['answer'])
        for letter, choice in zip(string.ascii_uppercase, choices):
            prompt += f"{letter}. {choice}\n"
        image_conversations += [
            {
                'from': "human",
                'value': prompt.strip()
            },
            {
                'from': "gpt",
                'value': f"{string.ascii_uppercase[correct_index]}. {qa_pair['answer']}"
            }
        ]
    if image_conversations:
        image_conversations[0]['value'] = "<image>\n" + image_conversations[0]['value']
        image_path = os.path.join("images", image_data['filename'])
        img = Image.open(os.path.join(DATASET_NAME, image_path))
        out_data.append({
            'id': image_data['image_id'],
            'image': image_path,
            'width': img.width,
            'height': img.height,
            'conversations': image_conversations
        })

# Pointing (bounding boxes)
print("Pointing QA")
with open(POINTING_FILE) as fp:
    pointing_data = json.load(fp)

# Get bounding box data
bbox_data = {
    bbox['box_id']: bbox
    for bbox in tqdm(pointing_data['boxes'], desc="Bounding boxes")
}

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

for image_data in tqdm(pointing_data['images'], desc="Pointing QA"):
    if image_data['split'] != "train":
        continue
    image_path = os.path.join("images", image_data['filename'])
    img = Image.open(os.path.join(DATASET_NAME, image_path))
    format_bbox = lambda bbox: "<box>[{}]</box>".format(normalize_coordinates(
        [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
        img.width, img.height
    ))

    image_conversations = []
    for qa_pair in image_data['qa_pairs']:
        prompt = f"Question: {qa_pair['question']}\nOptions:\n"
        choices = qa_pair['multiple_choices'] + [qa_pair['answer']]
        random.shuffle(choices)
        correct_index = choices.index(qa_pair['answer'])
        for letter, choice in zip(string.ascii_uppercase, choices):
            prompt += f"{letter}. {format_bbox(bbox_data[choice])}\n"
        image_conversations += [
            {
                'from': "human",
                'value': prompt.strip()
            },
            {
                'from': "gpt",
                'value': f"{string.ascii_uppercase[correct_index]}. {format_bbox(bbox_data[qa_pair['answer']])}"
            }
        ]
    if image_conversations:
        image_conversations[0]['value'] = "<image>\n" + image_conversations[0]['value']
        out_data.append({
            'id': image_data['image_id'],
            'image': image_path,
            'width': img.width,
            'height': img.height,
            'conversations': image_conversations
        })

with open(os.path.join(DATASET_NAME, "preprocessed.jsonl"), "w") as fp:
    for example in out_data:
        print(json.dumps(example), file=fp)
imgs_saved = len(os.listdir(os.path.join(DATASET_NAME, "images")))

print("Preprocessing complete! Saved as '{0}/preprocessed.jsonl' with {1} saved images in '{0}/images' and {2} total samples.".format(DATASET_NAME, imgs_saved, len(out_data)))

os.remove(IMAGE_ZIP)
os.remove(TELLING_ZIP)
os.remove(TELLING_FILE)
os.remove(POINTING_ZIP)
os.remove(POINTING_FILE)

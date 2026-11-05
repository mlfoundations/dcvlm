import json
import os
from PIL import Image
import random
from tqdm import tqdm
import wget
import zipfile

DATASET_NAME = "tallyqa"
print("Preparing dataset:", DATASET_NAME)

os.makedirs(os.path.join(DATASET_NAME, "images"), exist_ok=True)

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'
IMAGE_URL = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"]
IMAGE_ZIP = ["images.zip", "images2.zip"]
ANN_URL = "https://github.com/manoja328/TallyQA_dataset/raw/master/tallyqa.zip"
ANN_ZIP = "tallyqa.zip"
ANN_FILE = "train.json"

for image_url, image_zip in zip(IMAGE_URL, IMAGE_ZIP):
    print(f"Downloading images to '{image_zip}'")
    if not os.path.exists(image_zip):
        wget.download(image_url, out=image_zip)
    print(f"Extracting images to '{DATASET_NAME}/images'")
    with zipfile.ZipFile(image_zip) as zfp:
        zfp.extractall(os.path.join(DATASET_NAME, "images"))
print("Downloading annotations")
if not os.path.exists(ANN_FILE):
    wget.download(ANN_URL, out=ANN_ZIP)
    with zipfile.ZipFile(ANN_ZIP) as zfp:
        zfp.extract(ANN_FILE)

# Preprocessing

print("Starting preprocessing")
random.seed(0)

out_data = []

# Telling (normal QA)
with open(ANN_FILE) as fp:
    all_data = json.load(fp)

for question_data in tqdm(all_data):
    image_conversations = [
        {
            'from': "human",
            'value': question_data['question']
        },
        {
            'from': "gpt",
            'value': f"{question_data['answer']}"
        }
    ]
    if random.randrange(2) == 0:
        image_conversations[0]['value'] = "<image>\n" + image_conversations[0]['value']
    else:
        image_conversations[0]['value'] = image_conversations[0]['value'] + "\n<image>"

    # Resolve VG or COCO
    if question_data['image'].startswith("VG_100K"):
        image_path = os.path.join("images", question_data['image'])
        assert os.path.exists(os.path.join(DATASET_NAME, image_path))
    else:
        coco_split = os.path.dirname(question_data['image'])
        image_path = os.path.join("images", coco_split, question_data['image'])
        assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image_path))

    out_data.append({
        'id': str(question_data['question_id']),
        'image': image_path,
        'conversations': image_conversations
    })

with open(os.path.join(DATASET_NAME, "preprocessed.jsonl"), "w") as fp:
    for example in out_data:
        print(json.dumps(example), file=fp)

print("Preprocessing complete! Saved as '{0}/preprocessed.jsonl' with images in '{0}/images' and {1} total samples.".format(DATASET_NAME, len(out_data)))

for image_zip in IMAGE_ZIP:
    os.remove(image_zip)
os.remove(ANN_ZIP)
os.remove(ANN_FILE)
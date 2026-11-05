import csv
import gdown
import json
import os
import random
import tarfile
from tqdm import tqdm


DATASET_NAME = "art500k"
OUTPUT_DIR = "./{}".format(DATASET_NAME)
print("Preparing dataset:", DATASET_NAME)

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

IMAGE_URL = "1sWqPDINj7xSBaYmYxOsaYOLBwJZvg4BB" # GDrive ID
IMAGE_TAR = os.path.join(OUTPUT_DIR, "ART500K.tar.gz")
ANN_URL = "1HpEiuC7cQRBu0ykpbEH0iKb1Hh5rrvtE" # GDrive ID
ANN_FILE = os.path.join(OUTPUT_DIR, "ART500K_metadata.csv")

print(f"Downloading images to '{IMAGE_TAR}'")
gdown.download(id=IMAGE_URL, output=IMAGE_TAR, quiet=False)
print(f"Extracting images to '{DATASET_NAME}/images'")
with tarfile.open(IMAGE_TAR, "r:gz") as tfp:
    tfp.extractall(os.path.join(OUTPUT_DIR, "images"))
print("Downloading annotations")
gdown.download(id=ANN_URL, output=ANN_FILE, quiet=False)

COLUMNS = [
    'author_name', 'painting_name', 'image_url', 'Genre', 'Style', 'Nationality', 'Painting_School', 'Art_Movement', 'Field',
    'Date', 'Influenced_by', 'Media', 'Influenced_on', 'Family_and_Relatives', 'Tag', 'Pupils', 'Location', 'Original_Title',
    'Dimensions', 'Series', 'Teachers', 'Friends_and_Co_workers', 'Art_institution', 'Period', 'Theme', 'Path'
]
DESCRIPTION_PROMPTS = {
  "author_name": {"question": "Who created this painting?", "answer": "This work was painted by {}."},
  "painting_name": {"question": "What is the title of this painting?", "answer": "The painting is titled '{}'."},
  "Genre": {"question": "What genre does this painting belong to?", "answer": "It falls under the genre of {}."},
  "Style": {"question": "What is the style of this painting?", "answer": "The style of this painting is {}."},
  "Art_Movement": {"question": "Which art movement is this painting associated with?", "answer": "This painting is associated with the {} movement."},
  "Date": {"question": "When was this painting created?", "answer": "It was created in {}."},
  "Art_institution": {"question": "Where is this painting currently held?", "answer": "This painting is currently housed in {}."},
  "Period": {"question": "To which period does this painting belong?", "answer": "This piece is from the {} period."},
  "Theme": {"question": "What is the main theme of this painting?", "answer": "The central theme of the painting is {}."}
}
"""
Description prompts are generated using ChatGPT with this prompt:
```
For the following set of metadata about a painting:
['author_name', 'painting_name', 'Genre', 'Style', 'Art Movement', 'Date', 'Art institution', 'Period', 'Theme']
For each column, create a question asking about that information and an answer template with the corresponding information, such as:
{"question": "Who created this painting?", "answer": "This work was painted by {author_name}"}
```
"""

# Preprocessing

print("Starting preprocessing")
random.seed(0)

out_data = []

with open(ANN_FILE, newline="") as fp:
    all_data = [
        dict(zip(COLUMNS, line, strict=True))
        for line in csv.reader(fp)
    ]

for image_data in tqdm(all_data):
    image_data['painting_name'] = image_data['painting_name'].split("##", 1)[0]
    key_candidates = [
        key
        for key in DESCRIPTION_PROMPTS.keys()
        if image_data.get(key)
    ]
    chosen_key = random.choice(key_candidates)

    image_conversations = [
        {
            'from': "human",
            'value': DESCRIPTION_PROMPTS[chosen_key]['question']
        },
        {
            'from': "gpt",
            'value': DESCRIPTION_PROMPTS[chosen_key]['answer'].format(image_data[chosen_key])
        }
    ]
    if random.randrange(2) == 0:
        image_conversations[0]['value'] = "<image>\n" + image_conversations[0]['value']
    else:
        image_conversations[0]['value'] = image_conversations[0]['value'] + "\n<image>"

    image_path = os.path.join("images", image_data['Path'])

    assert os.path.exists(os.path.join(OUTPUT_DIR, image_path)), "Image: {} does not exist!!!".format(image_path)

    out_data.append({
        'id': image_data['painting_name'],
        'image': image_path,
        'conversations': image_conversations
    })

with open(os.path.join(OUTPUT_DIR, "preprocessed.jsonl"), "w") as fp:
    for example in out_data:
        print(json.dumps(example), file=fp)

print("Preprocessing complete! Saved as '{0}/preprocessed.jsonl' with images in '{0}/images' and {1} total samples.".format(DATASET_NAME, len(out_data)))

os.remove(IMAGE_TAR)
os.remove(ANN_FILE)

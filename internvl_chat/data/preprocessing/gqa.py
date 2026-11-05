import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import requests
import zipfile

GQA_QA_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"

output_dir = './gqa'
os.makedirs(output_dir, exist_ok=True)

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")

# Download and extract images
questions_path = os.path.join(output_dir, "questions.zip")
print('Downloading questions zip!')
download_file(GQA_QA_URL, questions_path)
unzip_file(questions_path, output_dir)

images_path = os.path.join(output_dir, "images.zip")
print('Downloading image zip!')
download_file(GQA_IMAGES_URL, images_path)
unzip_file(images_path, output_dir)

# Remove unnecessary files
os.remove("./gqa/challenge_all_questions.json")
os.remove("./gqa/challenge_balanced_questions.json")
os.remove("./gqa/readme.txt")
os.remove("./gqa/submission_all_questions.json")
os.remove("./gqa/test_all_questions.json")
os.remove("./gqa/test_balanced_questions.json")
os.remove("./gqa/testdev_all_questions.json")
os.remove("./gqa/testdev_balanced_questions.json")
os.remove("./gqa/val_all_questions.json")
os.remove("./gqa/val_balanced_questions.json")
shutil.rmtree("./gqa/train_all_questions")

new_jsonl_entries = []

with open("./gqa/train_balanced_questions.json") as f:
    json_contents = json.load(f)

for index, (key, content) in tqdm(enumerate(json_contents.items()), ascii=True, total=len(json_contents)):
    q_ = content["question"]
    a_ = content["fullAnswer"]
    img_id = content["imageId"]

    assert q_ is not None, "question is None!!!"
    assert a_ is not None, "answer is None!!!"

    img_path = os.path.join("images", "{}.jpg".format(img_id))
    assert os.path.exists(os.path.join(output_dir, img_path)), "Image does not exist!!!"

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(q_)}, {"from":"gpt", "value":a_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(q_)}, {"from":"gpt", "value":a_}]
    entry_ = {"id":len(new_jsonl_entries), "image":img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./gqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove('./gqa/train_balanced_questions.json')

print("Preprocessing complete! Saved as 'gqa/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
import json
import os
import io
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import shutil
import tarfile
import requests

JSONL_URL = 'https://huggingface.co/datasets/MMInstruction/ArxivQA/resolve/main/arxivqa.jsonl'
IMAGES_URL = 'https://huggingface.co/datasets/MMInstruction/ArxivQA/resolve/main/images.tgz'
LAB_MAPPING_UP = {i:chr(65+i) for i in range(26)}
LAB_MAPPING_DOWN = {i:chr(97+i) for i in range(26)}

###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

###### Download the files and unzip

output_dir = "./arxivqa"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

# # Download jsonl
train_jsonl_path = os.path.join(output_dir, "train.jsonl")
download_file(JSONL_URL, train_jsonl_path)

# Download and extract images.tgz
images_tar_path = os.path.join(output_dir, "images.tgz")
download_file(IMAGES_URL, images_tar_path)

# Extract the tar.gz file
print('Extracting folder...')
with tarfile.open(images_tar_path, 'r:gz') as tar:
    tar.extractall(output_dir)
    print(f"Extracted images to {output_dir}")

# Optionally remove the tar file after extraction
os.remove(images_tar_path)

###### Read and create preprocessed jsonl file

with open(train_jsonl_path) as f:
    json_contents = [json.loads(x) for x in f.readlines()]

new_jsonl_entries = []

# add entry to new jsonl list
for index, t in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):

    image = t['image']
    options = t['options']
    question = t['question']
    answer = t['label']
    rationale = t['rationale']

    assert os.path.exists(os.path.join(output_dir, image)), 'Image {} does not exist!!!'.format(image)

    prompt_ = ''
    assert question is not None, 'question is None!'
    prompt_ += 'Question: {}\n'.format(question)
    assert options is not None, 'question is mcq style, but options is None!!!'
    prompt_ += 'Options:\n'
    for indd, o in enumerate(options):
        if o[:2] not in ['{}.'.format(LAB_MAPPING_UP[indd]), '{})'.format(LAB_MAPPING_UP[indd]), '{}.'.format(LAB_MAPPING_DOWN[indd]), '{})'.format(LAB_MAPPING_DOWN[indd])]:
            prompt_ += '{}. {}\n'.format(LAB_MAPPING_UP[indd], o)
        else:
            prompt_ += '{}\n'.format(o)

    assert rationale is not None, 'rationale is None!!!'
    assert answer is not None, 'answer is None!!!'

    answer_ = '{}\nFinal Answer: {}'.format(rationale, answer)

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":answer_}]
    entry_ = {"id":len(new_jsonl_entries), "image": image, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./arxivqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'arxivqa/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

os.remove('./arxivqa/train.jsonl')
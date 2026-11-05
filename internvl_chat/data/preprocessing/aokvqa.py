import json
import os
import io
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import shutil
import tarfile
import requests

AOKVQA_URL = 'https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz'
LAB_MAPPING = {i:chr(97+i) for i in range(26)}
assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

# Function to get coco images path
def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

###### Download the files and unzip

output_dir = "./aokvqa"
os.makedirs(output_dir, exist_ok=True)

# Download
train_path = os.path.join(output_dir, "train.tar.gz")
download_file(AOKVQA_URL, train_path)

# Extract the tar.gz file
print('Extracting folder...')
with tarfile.open(train_path, 'r:gz') as tar:
    tar.extractall(output_dir)
    print(f"Extracted files to {output_dir}")

# Optionally remove the tar file after extraction
os.remove(train_path)

###### Read and create preprocessed jsonl file

train_json_path = os.path.join(output_dir, 'aokvqa_v1p0_train.json')

with open(train_json_path) as f:
    json_contents = json.load(f)

new_jsonl_entries = []

# add entry to new jsonl list
for index, t in enumerate(json_contents):

    question = t['question']
    choices = t['choices']
    correct_answer = choices[t['correct_choice_idx']]
    rationales = t['rationales']
    if isinstance(rationales, list):
        rationale = random.choice(rationales)
    else:
        rationale = rationales
    image_path = get_coco_path('train', t['image_id'], 'images/train2017')
    
    assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image_path)), 'Image {} does not exist!!!'.format(image_path)

    prompt_ = ''
    assert question is not None, 'question is None!'
    prompt_ += 'Question: {}\n'.format(question)
    assert choices is not None, 'question is mcq style, but options is None!!!'
    prompt_ += 'Options:\n'
    for indd, o in enumerate(choices):
        prompt_ += '{}. {}\n'.format(LAB_MAPPING[indd], o)

    assert rationale is not None, 'rationale is None!!!'
    assert correct_answer is not None, 'answer is None!!!'

    answer_ = '{}\nFinal Answer: {}'.format(rationale, correct_answer)

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":answer_}]
    entry_ = {"id":len(new_jsonl_entries), "image": image_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./aokvqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove('./aokvqa/aokvqa_v1p0_test.json')
os.remove('./aokvqa/aokvqa_v1p0_val.json')
os.remove('./aokvqa/specialized_vocab_train.csv')
os.remove('./aokvqa/aokvqa_v1p0_train.json')
os.remove('./aokvqa/large_vocab_train.csv')

print("Preprocessing complete! Saved as 'aokvqa/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
import json
import os
import shutil
import requests
import zipfile
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'
OKVQA_TRAIN_ANSWERS_URL = "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip"
OKVQA_TRAIN_ANSWERS = "mscoco_train2014_annotations.json"
OKVQA_TRAIN_QUESTIONS_URL = "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip"
OKVQA_TRAIN_QUESTIONS = "OpenEnded_mscoco_train2014_questions.json"
ROOT_DIR = './okvqa'

######## All utility functions

# Function to get coco images path
def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2014", f"COCO_train2014_{image_id:012}.jpg")


# Function to unzip files
def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")


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


######## first download the train Q & A pairs if they haven't been downloaded already

# create dataset directory if it does not exist
os.makedirs(ROOT_DIR, exist_ok=True)

# check if the original zip files have already been downloaded/failed to be erased to avoid duplicated downloads
upath_ = ROOT_DIR

dpath_ = os.path.join(upath_, 'mscoco_train2014_annotations.json.zip')
if not os.path.exists(dpath_):
    print('Downloading: {}'.format(OKVQA_TRAIN_ANSWERS))
    download_file(OKVQA_TRAIN_ANSWERS_URL, dpath_)
else:
    print(f"File '{dpath_}' already exists. Skipping download.")

if not os.path.exists(os.path.join(upath_, OKVQA_TRAIN_ANSWERS)):
    unzip_file(dpath_, upath_)
    print('Cleaned up and ready in: {}!'.format(upath_))
else:
    print(f"File '{upath_+OKVQA_TRAIN_ANSWERS}' already exists. Skipping unzip.")


dpath_ = os.path.join(upath_, 'OpenEnded_mscoco_train2014_questions.json.zip')
if not os.path.exists(dpath_):
    print('Downloading: {}'.format(OKVQA_TRAIN_QUESTIONS))
    download_file(OKVQA_TRAIN_QUESTIONS_URL, dpath_)
else:
    print(f"File '{dpath_}' already exists. Skipping download.")

if not os.path.exists(os.path.join(upath_, OKVQA_TRAIN_QUESTIONS)):
    unzip_file(dpath_, upath_)
    print('Cleaned up and ready in: {}!'.format(upath_))
else:
    print(f"File '{upath_+OKVQA_TRAIN_QUESTIONS}' already exists. Skipping unzip.")


######## We now want to start the reformatting procedure

# Load the question and answer annotations in json format
okvqa_train_questions = json.load(open(os.path.join(ROOT_DIR, OKVQA_TRAIN_QUESTIONS), 'r'))['questions']
okvqa_train_answers = json.load(open(os.path.join(ROOT_DIR, OKVQA_TRAIN_ANSWERS), 'r'))['annotations']

# Create a dictionary from image_ids to answers
okvqa_train_answers_dict = dict()
for index, answer in tqdm(enumerate(okvqa_train_answers), ascii=True, total=len(okvqa_train_answers)):
    okvqa_train_answers_dict[answer['image_id']] = answer['answers']


new_jsonl_entries = []

for index, question in tqdm(enumerate(okvqa_train_questions), ascii=True, total=len(okvqa_train_questions)):
    image_path = get_coco_path('train', question['image_id'], 'images/train2014')
    assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image_path)), 'Image {} does not exist!!!'.format(image_path)
    
    # The prompt will simply be the question
    prompt_ = question['question']

    # The response will be the first of the valid answers, since some of the later answers are noisy and/or wrong
    response_ = okvqa_train_answers_dict[question['image_id']][0]['answer']
    #response_ = random.choice(okvqa_train_answers_dict[question['image_id']])['answer']
    
     # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    if image_beginning:
        convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    
    entry_ = {"id": len(new_jsonl_entries), "image": image_path, "conversations": convs}
    new_jsonl_entries.append(entry_)


with open('./okvqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove(os.path.join(ROOT_DIR, OKVQA_TRAIN_QUESTIONS))
os.remove(os.path.join(ROOT_DIR, OKVQA_TRAIN_ANSWERS))
shutil.rmtree(os.path.join(ROOT_DIR, '__MACOSX'))


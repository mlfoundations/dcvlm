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

ZIP_URL = 'https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip'

output_dir = "./iconqa"
os.makedirs(output_dir, exist_ok=True)

###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

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

######## first download and unzip files

train_zip_file = os.path.join(output_dir, "train.zip")
print("Downloading zip...")
download_file(ZIP_URL, train_zip_file)
unzip_file(train_zip_file, output_dir)

# delete unnecessary files
print("Deleting unnecessary files!")
os.remove(os.path.join(output_dir, "iconqa_data", "LICENSE.md"))
os.remove(os.path.join(output_dir, "iconqa_data", "metadata.json"))
os.remove(os.path.join(output_dir, "iconqa_data", "pid_splits.json"))
os.remove(os.path.join(output_dir, "iconqa_data", "pid2skills.json"))
os.remove(os.path.join(output_dir, "iconqa_data", "problem_ids.json"))
os.remove(os.path.join(output_dir, "iconqa_data", "problems.json"))
os.remove(os.path.join(output_dir, "iconqa_data", "skills.json"))
shutil.rmtree(os.path.join(output_dir, "iconqa_data", "iconqa", "test"))
shutil.rmtree(os.path.join(output_dir, "iconqa_data", "iconqa", "val"))

# rename folder
os.rename(os.path.join(output_dir, "iconqa_data"), os.path.join(output_dir, "images"))

######## start the preprocessing

# first start processing fill_in_the_blank
new_jsonl_entries = []

root = os.path.join(output_dir, "images", "iconqa", "train", "fill_in_blank")
all_q_folders = os.listdir(root)

for q_index in tqdm(all_q_folders, ascii=True, total=len(all_q_folders)):
    files_ = os.listdir(os.path.join(root, q_index))
    assert len(files_) == 2, 'There are more than 2 files!!!'
    assert 'image.png' in files_, 'No image in files!!!'
    assert 'data.json' in files_, 'No question data in files!!!'

    # read json file
    with open(os.path.join(root, q_index, "data.json")) as f:
        contents = json.load(f)

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(contents["question"])}, {"from":"gpt", "value":contents["answer"]}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(contents["question"])}, {"from":"gpt", "value":contents["answer"]}]
    img_path = os.path.join("images", "iconqa", "train", "fill_in_blank", q_index, "image.png")
    assert os.path.exists(os.path.join(output_dir, img_path)), 'image path: {} does not exist!!!'.format(img_path)
    entry_ = {"id":len(new_jsonl_entries), "image": img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)


# next start processing choose_img
root = os.path.join(output_dir, "images", "iconqa", "train", "choose_img")
all_q_folders = os.listdir(root)

for q_index in tqdm(all_q_folders, ascii=True, total=len(all_q_folders)):
    files_ = os.listdir(os.path.join(root, q_index))
    assert 'image.png' in files_, 'No image in files!!!'
    assert 'data.json' in files_, 'No question data in files!!!'

    image_list = []
    for f_ in files_:
        if '.png' in f_:
            if f_ != 'image.png':
                image_list.append(f_)

    image_list.sort()
    image_list = ['image.png'] + image_list

    for ind in range(len(image_list)):
        image_list[ind] = os.path.join("images", "iconqa", "train", "choose_img", q_index, image_list[ind])

    # read json file
    with open(os.path.join(root, q_index, "data.json")) as f:
        contents = json.load(f)

    prompt_ = ''
    prompt_ += '<image>Question: {}\n'.format(contents["question"])
    prompt_ += 'Options:\n'
    for indd, _ in enumerate(contents["choices"]):
        prompt_ += '{}. {}\n'.format(indd+1, '<image>')

    answer_ = contents["answer"] + 1

    assert prompt_.count("<image>") == len(image_list)
    for img_ in image_list:
        assert os.path.exists(os.path.join(output_dir, img_)), 'image path: {} does not exist!!!'.format(img_)

    convs = [{"from":"human", "value":"{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    entry_ = {"id":len(new_jsonl_entries), "image": image_list, "conversations": convs}
    new_jsonl_entries.append(entry_)



# next start processing choose_txt
root = os.path.join(output_dir, "images", "iconqa", "train", "choose_txt")
all_q_folders = os.listdir(root)

for q_index in tqdm(all_q_folders, ascii=True, total=len(all_q_folders)):
    files_ = os.listdir(os.path.join(root, q_index))
    assert len(files_) == 2, 'There are more than 2 files!!!'
    assert 'image.png' in files_, 'No image in files!!!'
    assert 'data.json' in files_, 'No question data in files!!!'

    # read json file
    with open(os.path.join(root, q_index, "data.json")) as f:
        contents = json.load(f)

    prompt_ = ''
    prompt_ += 'Question: {}\n'.format(contents["question"])
    prompt_ += 'Options:\n'
    for indd, c in enumerate(contents["choices"]):
        prompt_ += '{}. {}\n'.format(indd+1, c)

    answer_ = contents["choices"][contents["answer"]]

    img_path = os.path.join("images", "iconqa", "train", "choose_txt", q_index, "image.png")
    assert os.path.exists(os.path.join(output_dir, img_path)), 'image path: {} does not exist!!!'.format(img_path)

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":answer_}]
    entry_ = {"id":len(new_jsonl_entries), "image": img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)


with open('./iconqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'iconqa/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

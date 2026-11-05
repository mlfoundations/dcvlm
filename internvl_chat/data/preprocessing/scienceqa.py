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

TRAIN_PARQUET_URL = 'https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/main/data/train-00000-of-00001-1028f23e353fbe3e.parquet'
LAB_MAPPING = {i:chr(97+i) for i in range(26)}

output_dir = "./scienceqa"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

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

train_parquet_path = os.path.join(output_dir, "train.parquet")
download_file(TRAIN_PARQUET_URL, train_parquet_path)

###### Read and create preprocessed jsonl file

new_jsonl_entries = []
imgs_saved = 0

df = pd.read_parquet(train_parquet_path)
for sample_index, row in tqdm(df.iterrows(), ascii=True):
    img_bytes = row["image"]
    hint = row["hint"]
    question = row["question"]
    choices = row["choices"]
    answer = row["answer"]
    solution = row["solution"]

    file_name, image, width, height = None, None, None, None
    if img_bytes:
        image = Image.open(io.BytesIO(img_bytes['bytes']))
        file_name = os.path.join(output_dir, 'images', 'q{}_image.png'.format(sample_index))
        width, height = image.size
        image.save(file_name)
        imgs_saved += 1

    prompt_ = ''
    if hint is not None and isinstance(hint, str) and len(hint)>0 and hint!='':
        prompt_ += 'Hint: {}\n'.format(hint)
    assert question is not None, 'question is None!'
    prompt_ += 'Question: {}\n'.format(question)
    prompt_ += 'Options:\n'
    for indd, o in enumerate(choices):
        prompt_ += '{}. {}\n'.format(LAB_MAPPING[indd], o)
    
    answer_ = ''
    if solution is None:
        answer_ = 'Final Answer: {}'.format(choices[answer])
    else:    
        answer_ = '{}\nFinal Answer: {}'.format(solution, choices[answer])

    # add entry to new jsonl list
    if file_name:
        assert height is not None, 'Img height is None!!'
        assert width is not None, 'Img width is None!!'
        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":answer_}]
        entry_ = {"id":sample_index, "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)
    else:
        convs = [{"from":"human", "value":"{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
        entry_ = {"id":sample_index, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open('./scienceqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'scienceqa/preprocessed.jsonl' with {} images saved and {} total samples.".format(imgs_saved, len(new_jsonl_entries)))

os.remove(os.path.join(output_dir, 'train.parquet'))
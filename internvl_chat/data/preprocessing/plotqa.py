import os
import math
import gdown
import random
import tarfile
import json
import shutil
from tqdm import tqdm

random.seed(0)

OUTPUT_DIR = "./plotqa"
SHARD_SIZE = 4_000_000


def download_file_from_google_drive(url: str, destination: str):
    file_id = url.split('/')[5]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(download_url, destination, quiet=False)


os.makedirs(OUTPUT_DIR, exist_ok=True)
# Images
download_file_from_google_drive("https://drive.google.com/file/d/1AYuaPX-Lx7T0GZvnsPgN11Twq2FZbWXL/view?usp=sharing", f"{OUTPUT_DIR}/plotqa.tar.gz")

# Base version (8,190,674 QA pairs)
# download_file_from_google_drive("https://drive.google.com/file/d/1bBSUutd-Die27ZH3QTMVhBjW9l5hAwrr/view?usp=sharing", f"{OUTPUT_DIR}/qa_pairs.json")
# Extended version (28,952,641 QA pairs)
download_file_from_google_drive("https://drive.google.com/file/d/1UNvkdq1YJD_ne6D3zbWtoQij37AtfpNp/view?usp=sharing", f"{OUTPUT_DIR}/qa_pairs.json")
# Annotations (not needed)
# download_file_from_google_drive("https://drive.google.com/file/d/1VzWwxBVrlep17BGZU17GpLuGpwjyWbzq/view?usp=sharing", f"{OUTPUT_DIR}/annotations.json")

print("Extracting PlotQA.zip")
with tarfile.open(f"{OUTPUT_DIR}/plotqa.tar.gz") as images_tar:
    images_tar.extractall(OUTPUT_DIR)
shutil.move(f"{OUTPUT_DIR}/png", f"{OUTPUT_DIR}/images")


questions = []
with open(f"{OUTPUT_DIR}/qa_pairs.json", "r") as f:
    questions = json.load(f)['qa_pairs']

uid_set = set()
preprocessed = []
for entry in tqdm(questions, desc=f"Processing questions", total=len(questions)):
    query = entry['question_string']
    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    value_str = f"<image>{query}" if image_beginning else f"{query}<image>"
    convs = [
        {"from": "human", "value": value_str},
        {"from": "gpt", "value": entry["answer"]},
    ]
    image_path = os.path.join("images", f'{entry["image_index"]}.png')
    assert os.path.exists(
        os.path.join(OUTPUT_DIR, image_path)
    ), "Image in dataset does not exist on filesystem."
    if entry['question_id'] in uid_set:
        raise ValueError(f"Duplicate uid {entry['question_id']}")
    uid_set.add(entry['question_id'])
    preprocessed.append({
        "id": entry['question_id'],
        "image": image_path,
        "conversations": convs,
    })

os.remove(f"{OUTPUT_DIR}/plotqa.tar.gz")
os.remove(f"{OUTPUT_DIR}/qa_pairs.json")

num_shards = math.ceil(len(preprocessed) / SHARD_SIZE)

for shard_id in range(num_shards):
    with open(f"{OUTPUT_DIR}/preprocessed_{shard_id}.jsonl", "w") as f:
        start_idx = shard_id * SHARD_SIZE
        end_idx = min((shard_id + 1) * SHARD_SIZE, len(preprocessed))
        for entry in tqdm(preprocessed[start_idx:end_idx], desc=f"Writing shard {shard_id + 1}/{num_shards}", total=end_idx - start_idx):
            f.write(json.dumps(entry) + "\n")

print(f"Downloaded PlotQA dataset to {OUTPUT_DIR}")

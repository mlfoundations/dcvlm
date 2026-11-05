import os
import gdown
import random
import tarfile
import zipfile
import json
import shutil
from tqdm import tqdm

random.seed(0)

OUTPUT_DIR = "./lrv_instruction"

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

# Instruction
gdown.download("https://drive.google.com/uc?id=1k9MNV-ImEV9BYEOeLEIb4uGEUZjd3QbM", f"{OUTPUT_DIR}/image.tar.gz")
gdown.download("https://drive.google.com/uc?id=1pWkxE2kqpys1VdwBi99ZXN6-XY5SqhwU", f"{OUTPUT_DIR}/filter_cap1.json")
gdown.download("https://drive.google.com/uc?id=1NTxkuRPlvDn7aWaJpK_yb0p5r0cxPLNZ", f"{OUTPUT_DIR}/filter_cap_more1.json")

## Old version
# gdown.download("https://drive.google.com/uc?id=1E83vpCY8ofEDGFZLjpIjPWHHJvocrWVj", f"{OUTPUT_DIR}/filter_cap_more.json")
# gdown.download("https://drive.google.com/uc?id=1OlNxY1eC9Fg10LEcEo1xS6cyOWYera_G", f"{OUTPUT_DIR}/filter_cap.json")

# Chart
gdown.download("https://drive.google.com/uc?id=1Dey-undzW2Nl21CYLFSkP_Y4RrfRJkYd", f"{OUTPUT_DIR}/chart_image.zip")
gdown.download("https://drive.google.com/uc?id=13j2U-ectsYGR92r6J5hPdhT8T5ezItHF", f"{OUTPUT_DIR}/chat_release_update.json")

with tarfile.open(f"{OUTPUT_DIR}/image.tar.gz") as images_tar:
    images_tar.extractall(f"{OUTPUT_DIR}/instruction")
shutil.move(f"{OUTPUT_DIR}/instruction/image", f"{OUTPUT_DIR}/images/instruction")

with zipfile.ZipFile(f"{OUTPUT_DIR}/chart_image.zip") as images_zip:
    images_zip.extractall(f"{OUTPUT_DIR}/chart")
shutil.move(f"{OUTPUT_DIR}/chart/chart_image", f"{OUTPUT_DIR}/images/chart_image")

## Old version
# with open(f"{OUTPUT_DIR}/filter_cap.json", "r") as f:
#     filter_cap = json.load(f)
# with open(f"{OUTPUT_DIR}/filter_cap_more.json", "r") as f:
#     filter_cap_more = json.load(f)

with open(f"{OUTPUT_DIR}/filter_cap1.json", "r") as f:
    filter_cap1 = json.load(f)

with open(f"{OUTPUT_DIR}/filter_cap_more1.json", "r") as f:
    filter_cap_more1 = json.load(f)

with open(f"{OUTPUT_DIR}/chat_release_update.json", "r") as f:
    chat_release_update = json.load(f)


preprocessed, cnt = [], 0

# Instruction images
instruction_qs = filter_cap1 + filter_cap_more1
for entry in tqdm(instruction_qs, desc=f"Processing instruction questions", total=len(instruction_qs)):
    query = entry['question']
    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    value_str = f"<image>{query}" if image_beginning else f"{query}<image>"
    convs = [
        {"from": "human", "value": value_str},
        {"from": "gpt", "value": entry["answer"]},
    ]
    image_path = os.path.join("images", "instruction", f'{entry["image_id"]}.jpg')
    assert os.path.exists(
        os.path.join(OUTPUT_DIR, image_path)
    ), "Image in dataset does not exist on filesystem."
    preprocessed.append({
        "id": cnt,
        "image": image_path,
        "conversations": convs,
    })
    cnt += 1

# Chart images
for entry in tqdm(chat_release_update, desc=f"Processing chart questions", total=len(chat_release_update)):
    query = entry['question']
    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    value_str = f"<image>{query}" if image_beginning else f"{query}<image>"
    convs = [
        {"from": "human", "value": value_str},
        {"from": "gpt", "value": entry["answer"]},
    ]
    image_path = os.path.join("images", "chart_image", entry["image_id"])
    assert os.path.exists(
        os.path.join(OUTPUT_DIR, image_path)
    ), "Image in dataset does not exist on filesystem."
    preprocessed.append({
        "id": cnt,
        "image": image_path,
        "conversations": convs,
    })
    cnt += 1

with open(f"{OUTPUT_DIR}/preprocessed.jsonl", "w") as f:
    for entry in tqdm(preprocessed, desc="Writing to jsonl", total=len(preprocessed)):
        f.write(json.dumps(entry) + "\n")

os.remove(f"{OUTPUT_DIR}/image.tar.gz")
os.remove(f"{OUTPUT_DIR}/filter_cap1.json")
os.remove(f"{OUTPUT_DIR}/filter_cap_more1.json")
os.remove(f"{OUTPUT_DIR}/chat_release_update.json")
os.remove(f"{OUTPUT_DIR}/chart_image.zip")
shutil.rmtree(f"{OUTPUT_DIR}/chart")
shutil.rmtree(f"{OUTPUT_DIR}/instruction")

## Old version
# os.remove(f"{OUTPUT_DIR}/filter_cap.json")
# os.remove(f"{OUTPUT_DIR}/filter_cap_more.json")

print(f"Downloaded LRV dataset to {OUTPUT_DIR}")

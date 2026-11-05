###################################
# 1. download the mmc_instruct dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0)  # Ensure reproducibility
import json
import shutil
import tarfile

# Step 1: Download the dataset
snapshot_download(
    repo_id="xywang1/MMC",
    repo_type="dataset",
    allow_patterns=["MMC-Instruction/*"],
    local_dir="mmc_data"
)
output_dir = "./mmc_instruct/images"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
cnt = 0
imgs_saved = 0

image_files = ["mmc_instruction_arxiv_images.tar.gz", "mmc_instruction_non-arxiv_images.tar.gz"]
text_files = ["mmc_instruction_arxiv_text.jsonl", "mmc_instruction_non-arxiv_text.jsonl"]

def process_file(image_file, text_file):
    global imgs_saved, cnt
    new_imgs_saved = 0
    entries = []
    img_base = image_file.split('.tar.gz')[0]
    base_name = text_file.split('_text')[0]

    # Extract images into a dictionary
    image_file_path = os.path.join("mmc_data", "MMC-Instruction", image_file)
    img_dict = {}
    print("Untarring {}!".format(image_file_path))
    with tarfile.open(image_file_path, "r:gz") as tar:
        tms = tar.getmembers()
        for member in tqdm(tms, total=len(tms), ascii=True):
            if member.isfile():
                f = tar.extractfile(member)
                img_dict[member.name] = f.read()

    # Load text data
    text_file_path = os.path.join("mmc_data", "MMC-Instruction", text_file)
    df = pd.read_json(text_file_path, lines=True)

    for sample_index, row in tqdm(df.iterrows(), ascii=True, desc=f"Processing {text_file}", total=len(df)):
        image_id = os.path.join(img_base, row["image_id"])
        question = row["question"]
        answer = row["answer"]
        
        # Retrieve image bytes using image_id
        if image_id not in img_dict:
            print(f"Image ID {image_id} not found in {image_file}, skipping sample.")
            continue

        # Check if the image has already been saved
        img_data = img_dict[image_id]
        if isinstance(img_data, dict):
            file_name = img_data.get("file_name")
            width = img_data.get("width")
            height = img_data.get("height")
        else:
            img_bytes = img_data
            image = Image.open(io.BytesIO(img_bytes))
            file_name = f"{output_dir}/{base_name}_{new_imgs_saved}.png"
            width, height = image.size
            image.save(file_name)
            new_imgs_saved += 1

            img_dict[image_id] = {
                "file_name": file_name,
                "width": width,
                "height": height
            }

        # Create conversation
        image_beginning = random.choice([0, 1])
        if image_beginning:
            convs = [{"from":"human", "value": f"<image>{question}"}, {"from":"gpt", "value": answer}]
        else:
            convs = [{"from":"human", "value": f"{question}<image>"}, {"from":"gpt", "value": answer}]
        
        # Create JSON entry
        entry_ = {
            "id": cnt,
            "image": 'images/' + os.path.basename(file_name),
            "height": height,
            "width": width,
            "conversations": convs
        }
        entries.append(entry_)
        cnt += 1
    
    imgs_saved += new_imgs_saved
    
    return entries

# Step 2: Process the files, create JSONL entries and save images
for image_file, text_file in zip(image_files, text_files):
    new_jsonl_entries.extend(process_file(image_file, text_file))

# Step 3: Save preprocessed data as JSONL
with open('./mmc_instruct/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('mmc_data')

print(f"Preprocessing complete! Saved as 'mmc_instruct/preprocessed.jsonl' with {imgs_saved} saved images in {output_dir} and {cnt+1} total samples.")
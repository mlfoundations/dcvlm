from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="LIME-DATA/infovqa",
    repo_type="dataset",
    allow_patterns=["data/train*"],  
    local_dir="infovqa_parquets"
)

output_dir = "./infovqa/images"
os.makedirs(output_dir, exist_ok=True)

def avg_hash(img):
    img = img.resize((10, 10), Image.Resampling.LANCZOS)
    img = img.convert("L")
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)

    bits = "".join(["1" if pixel > avg else "0" for pixel in pixels])
    hex_hash = hex(int(bits, 2))[2:].rjust(16, "0")
    return hex_hash

new_jsonl_entries = []
n_images = 0
n_samples = 0
seen_hashes = {}

parquets = [x for x in os.listdir("./infovqa_parquets/data") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./infovqa_parquets/data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        img_dict = row["image"]
        question = row["question"]
        answer = row["answers"][0]

        # Save image if it's not a duplicate
        file_name, height, width = None, None, None
        img_bytes = img_dict["bytes"]
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        img_hash = avg_hash(image)  # Get image hash
        if img_hash in seen_hashes:
            file_name, height, width = seen_hashes[img_hash]
        else:
            file_name = f"{output_dir}/img_{n_images}.png"  # Generate a unique name
            width, height = image.size  # Get width and height
            image.save(file_name)  # Save the image
            seen_hashes[img_hash] = (file_name, height, width)

            n_images += 1

        # Add entry to new jsonl list
        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(question)}, {"from":"gpt", "value":answer}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(question)}, {"from":"gpt", "value":answer}]
        entry_ = {"id":sample_index, "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./infovqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('infovqa_parquets')

print("Preprocessing complete! Saved as 'infovqa/preprocessed.jsonl' with {} images in {} and {} total samples.".format(n_images, output_dir, n_samples))
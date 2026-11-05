###################################
# 1. download the scigraphqa dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
import zipfile

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil


# path where files are temporarily stored
DOWNLOAD_DIR = "scigraphqa_parquets"
# output dir
OUTPUT_DIR = "./scigraphqa"
# image output dir
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


def main():
    snapshot_download(
        repo_id="alexshengzhili/SciGraphQA-295K-train", repo_type="dataset", local_dir=DOWNLOAD_DIR
    )
    with zipfile.ZipFile(os.path.join(DOWNLOAD_DIR, "img.zip"), "r") as fp:
        fp.extractall(DOWNLOAD_DIR)

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    new_jsonl_entries = []
    cnt = -1
    imgs_saved = 0
    parquets = [
        x for x in sorted(os.listdir(os.path.join(DOWNLOAD_DIR, "data"))) if x.endswith(".parquet")
    ]
    unique_names = set()
    for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
        df = pd.read_parquet(os.path.join(DOWNLOAD_DIR, "data", parquet))
        for sample_index, row in tqdm(df.iterrows(), ascii=True):
            image_path = os.path.join(DOWNLOAD_DIR, "imgs", "train", row["image_file"])
            image = Image.open(image_path)
            width, height = image.size
            
            # unique_name
            image_output_name = os.path.join(
                OUTPUT_IMAGE_DIR, f"parquet_{parquet_index}_sample_{sample_index}.png"
            )
            assert image_output_name not in unique_names
            unique_names.add(image_output_name)

            #save image
            image.save(image_output_name)
            imgs_saved += 1
            cnt += 1 # each image has 1 task

            conversations = [turn for turn in row["conversations"]]

            # conversations are in the correct format
            entry_ = {
                "id": cnt,
                "image": os.path.join("images", os.path.basename(image_output_name)),
                "height": height,
                "width": width,
                "conversations": conversations
            }
            new_jsonl_entries.append(entry_)

    # store jsonl
    output_file = os.path.join(OUTPUT_DIR, "preprocessed.jsonl")
    with open(output_file, "w") as fp:
        for entry in new_jsonl_entries:
            fp.write(json.dumps(entry) + "\n")

    print(
        f"Preprocessing complete! Saved as {output_file} with {imgs_saved} saved images in {OUTPUT_IMAGE_DIR} and {cnt+1} total samples."
    )
    
    shutil.rmtree(DOWNLOAD_DIR)


if __name__ == "__main__":
    main()

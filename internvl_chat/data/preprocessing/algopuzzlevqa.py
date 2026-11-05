###################################
# 1. download the algopuzzlevqa dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import zipfile


# path where parquets are temporarily stored
DOWNLOAD_DIR = "algopuzzlevqa_parquets"
# output dir
OUTPUT_DIR = "./algopuzzlevqa"
# image output dir
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


def main():
    snapshot_download(
        repo_id="declare-lab/AlgoPuzzleVQA",
        revision="refs/convert/parquet",
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR,
    )
    parquet_file = os.path.join(DOWNLOAD_DIR, "default/data/0000.parquet")

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    new_jsonl_entries = []
    cnt = -1
    imgs_saved = 0

    df = pd.read_parquet(parquet_file)
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        img_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(img_bytes))  # bytes to image
        width, height = image.size

        # image unique name
        image_output_name = os.path.join(OUTPUT_IMAGE_DIR, f"sample_{sample_index}.png")

        # save image
        image.save(image_output_name)
        imgs_saved += 1
        cnt += 1  # each image has 1 task

        # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        image_beginning = random.choice([0, 1])
        if image_beginning:
            convs = [
                {"from": "human", "value": f"<image>{row['question']}"},
                {"from": "gpt", "value": row["answer"]},
            ]
        else:
            convs = [
                {"from": "human", "value": f"{row['question']}<image>"},
                {"from": "gpt", "value": row["answer"]},
            ]
        
        entry_ = {
            "id": cnt,
            "image": os.path.join("images", os.path.basename(image_output_name)),
            "height": height,
            "width": width,
            "conversations": convs,
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

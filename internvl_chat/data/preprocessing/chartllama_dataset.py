###################################
# 1. download the chartllama_dataset dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
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


# path where files are temporarily stored
DOWNLOAD_DIR = "chartllama_dataset_files"
# output dir
OUTPUT_DIR = "./chartllama_dataset"
# image output dir
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


def main():
    snapshot_download(
        repo_id="listen2you002/ChartLlama-Dataset",
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR,
    )
    # unzip images
    with zipfile.ZipFile(os.path.join(DOWNLOAD_DIR, "ours.zip"), "r") as fp:
        fp.extractall(DOWNLOAD_DIR)

    # collect all annotation files
    annotation_files = [x for x in sorted(os.listdir(DOWNLOAD_DIR)) if x.endswith(".json")]

    new_jsonl_entries = []
    cnt = -1
    imgs_saved = 0

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    set_image_paths = set()

    for i, annotation_file in tqdm(enumerate(annotation_files), ascii=True):
        with open(os.path.join(DOWNLOAD_DIR, annotation_file), "r") as fp:
            annotations = json.load(fp)

        for j, line in enumerate(annotations):
            # save image
            image_path = os.path.join(DOWNLOAD_DIR, line["image"])
            image = Image.open(image_path)
            width, height = image.size

            # image unique name
            image_output_name = os.path.join(
                OUTPUT_IMAGE_DIR, f"annotation_file_{i}_sample_{j}.png"
            )
            assert (
                image_output_name not in set_image_paths
            ), f"Duplicate mage path {image_output_name}."
            set_image_paths.add(image_output_name)

            # save image
            image.save(image_output_name)
            imgs_saved += 1
            cnt += 1  # each image has 1 task

            # <image> is already randomly positioned
            # conversations are already in the correct format
            entry_ = {
                "id": cnt,
                "image": os.path.join("images", os.path.basename(image_output_name)),
                "height": height,
                "width": width,
                "conversations": line["conversations"],
            }
            new_jsonl_entries.append(entry_)

    # store
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

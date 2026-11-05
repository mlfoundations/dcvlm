###################################
# 1. download the ChartX dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
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
DOWNLOAD_DIR = "chartx_files"
# output dir
OUTPUT_DIR = "./chartx"
# image output dir
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


def main():
    snapshot_download(repo_id="U4R/ChartX", repo_type="dataset", local_dir=DOWNLOAD_DIR)
    with zipfile.ZipFile(os.path.join(DOWNLOAD_DIR, "ChartX_png.zip"), "r") as fp:
        fp.extractall(DOWNLOAD_DIR)

    new_jsonl_entries = []
    cnt = -1
    imgs_saved = 0

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    annotation_file = "ChartX_annotation_val.json"
    # annotations are in json format
    with open(os.path.join(DOWNLOAD_DIR, annotation_file), "r") as fp:
        annotations = json.load(fp)

    for i, line in enumerate(tqdm(annotations, ascii=True)):
        # save image
        image_path = os.path.join(DOWNLOAD_DIR, "ChartX_png", line["img"].replace("./", ""))
        image = Image.open(image_path)
        # unique name
        image_output_name = os.path.join(OUTPUT_IMAGE_DIR, f"sample_{i}.png")
        width, height = image.size
        # save image
        image.save(image_output_name)
        imgs_saved += 1

        # 4 possible tasks (not all images have all 4)
        for task in ["description", "summarization", "redrawing", "QA"]:
            sample = line.get(task, None)
            # check if image has certain task
            if sample is not None:
                cnt += 1

                # some output keys are called ouput_put (idk why)
                output_key = "output" if "output" in sample else "ouput_put"

                # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                image_beginning = random.choice([0, 1])
                if image_beginning:
                    convs = [
                        {"from": "human", "value": f"<image>{sample['input']}"},
                        {"from": "gpt", "value": sample[output_key]},
                    ]
                else:
                    convs = [
                        {"from": "human", "value": f"{sample['input']}<image>"},
                        {"from": "gpt", "value": sample[output_key]},
                    ]
                entry_ = {
                    "id": cnt,
                    "image": os.path.join("images", os.path.basename(image_output_name)),
                    "height": height,
                    "width": width,
                    "conversations": convs,
                }
                new_jsonl_entries.append(entry_)

    # store
    output_file = os.path.join(OUTPUT_DIR, f"preprocessed.jsonl")
    with open(output_file, "w") as fp:
        for entry in new_jsonl_entries:
            fp.write(json.dumps(entry) + "\n")

    print(
        f"Preprocessing complete! Saved as {output_file} with {imgs_saved} saved images in {OUTPUT_IMAGE_DIR} and {cnt+1} total samples."
    )

    shutil.rmtree(DOWNLOAD_DIR)


if __name__ == "__main__":
    main()

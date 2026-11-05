import argparse
from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import gzip
import shutil
import subprocess

# Get filename and remove extension
DS_NAME = os.path.basename(__file__).split(".")[0] 
# Temporary this for download
TEMP_DIR = f"./{DS_NAME}_temp"
# Final directory for the images
IMAGES_DIR = f"./{DS_NAME}/images"
# Final path for annotations
ANNOTATIONS_FILE = lambda idx: f"./{DS_NAME}/preprocessed_{idx}.jsonl"

# 2gb per file
SPLIT_SIZE = 1000000


def main():

    snapshot_download(repo_id="SirNeural/flan_v2", repo_type="dataset", local_dir=TEMP_DIR)

    if not "flan_fs_opt_train.jsonl.gz" in os.listdir(TEMP_DIR):
        subprocess.run(f"cat {TEMP_DIR}/flan_fs_opt_train_part1.jsonl.gz {TEMP_DIR}/flan_fs_opt_train_part2.jsonl.gz {TEMP_DIR}/flan_fs_opt_train_part3.jsonl.gz > {TEMP_DIR}/flan_fs_opt_train.jsonl.gz", shell=True)
        os.remove(f"{TEMP_DIR}/flan_fs_opt_train_part1.jsonl.gz")
        os.remove(f"{TEMP_DIR}/flan_fs_opt_train_part2.jsonl.gz")
        os.remove(f"{TEMP_DIR}/flan_fs_opt_train_part1.jsonl.gz")
    else:
        print("flan_fs_opt already merged")

    os.makedirs(f"./{DS_NAME}", exist_ok=True)

    # recursively iterate all directories to retrieve all .jsonl.gz files
    annotations = [
        os.path.join(root, x)
        for root, dirs, files in os.walk(TEMP_DIR)
        for x in files
        if x.endswith("train.jsonl.gz")
    ]

    new_jsonl_annotation = []
    split_counter = 0
    split_idx = 0

    start_from_annot = ""
    start_from_split_idx = 0
  
    if start_from_split_idx and start_from_annot:
        aidx = annotations.index(start_from_annot)
        annotations = annotations[aidx:]
        split_idx = start_from_split_idx

    print("Files to be processed -> {}".format(annotations))


    delete_temp_dir = True

    # iterate over jsonl.gz and open each with gzip
    for annotation_idx, annotation_path in enumerate(annotations):
        print(f"Processing {annotation_path}...", flush=True)
        annotation_samples = gzip.open(annotation_path)
        
        
        try:
            for sample in annotation_samples:
            
                sample = eval(sample)

                # FLAN is text-only, thus retrieve inputs and targets
                conv = [
                    {"from": "human", "value": sample["inputs"]},
                    {"from": "gpt", "value": sample["targets"]},
                ]
                entry = {"id": split_counter, "conversations": conv}
                new_jsonl_annotation.append(entry)
                split_counter += 1

                # Save annotation file every SPLIT_SIZE samples
                if split_counter >= SPLIT_SIZE:
                    with open(ANNOTATIONS_FILE(split_idx), 'w') as f:
                        for entry in new_jsonl_annotation:
                            f.write(json.dumps(entry) + '\n')
                    
                    print(
                        f"Preprocessing complete! Saved as '{ANNOTATIONS_FILE(split_idx)}' with 0 saved images and {split_counter} total samples.",
                        flush=True
                    )

                    split_counter = 0
                    split_idx += 1
                    new_jsonl_annotation = []
        except:
            # Unhandled error --> broken files on the flanv2 repo
            pass
            delete_temp_dir = False
        finally:
            # Save what you have and move to the next annotation
            with open(ANNOTATIONS_FILE(split_idx), 'w') as f:
                for entry in new_jsonl_annotation:
                    f.write(json.dumps(entry) + '\n')
                    
                print(
                    f"Broken Annotation! Saved as '{ANNOTATIONS_FILE(split_idx)}' with 0 saved images and {split_counter} total samples.",
                    flush=True
                )

                split_counter = 0
                split_idx += 1
                new_jsonl_annotation = []

            

    if delete_temp_dir:
        shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    main()
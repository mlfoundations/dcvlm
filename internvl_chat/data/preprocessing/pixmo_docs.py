"""
pixmo_docs.py

This script downloads the PixMo-Docs dataset from Hugging Face using snapshot_download,
then processes all four subsets ("charts", "diagrams", "tables", "other") by combining
their "train-*.parquet" shards. The processed output is stored in the following structure:

pixmo_docs/
├── images/
│    ├── charts_0.png
│    ├── charts_1.png
│    ├── ...
│    ├── diagrams_0.png
│    ├── diagrams_1.png
│    ├── ...
│    ├── tables_0.png
│    ├── tables_1.png
│    ├── ...
│    ├── other_0.png
│    ├── other_1.png
│    └── ...
├── charts_preprocessed.jsonl
├── diagrams_preprocessed.jsonl
├── tables_preprocessed.jsonl
└── other_preprocessed.jsonl

Each JSONL file is in an InternVL-Chat style, e.g.:
{
  "id": 0,
  "image": "images/charts_0.png",
  "height": 960,
  "width": 1280,
  "conversations": [
    {"from": "human", "value": "<image>What is in this chart?"},
    {"from": "gpt", "value": "It shows the distribution ..."},
    {"from": "human", "value": "How many categories are displayed?"},
    {"from": "gpt", "value": "Four main ones ..."}
  ]
}

Usage:
  python pixmo_docs.py
No additional arguments are required.
"""

import os
import json
import glob
import random
import concurrent.futures
from tqdm import tqdm
from PIL import Image

from huggingface_hub import snapshot_download
from datasets import load_dataset
import shutil

THREAD_WORKERS = 8
random.seed(0)

SUBSETS = ["charts", "diagrams", "tables", "other"]

def process_and_save_image(entry, subset_name, idx, images_dir):
    """
    Saves the PIL image to the output directory and creates the final
    conversation-based data entry in the required format.
    """
    pil_img = entry["image"]
    width, height = pil_img.size

    # Construct a filename to avoid collisions across subsets
    filename = f"{subset_name}_{idx}.png"
    file_path = os.path.join(images_dir, filename)

    # Save the image
    pil_img.save(file_path)

    # Build the conversation from the multiple Q/A pairs
    questions = entry["questions"]["question"]
    answers   = entry["questions"]["answer"]

    convs = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        if i == 0:
            # Randomly place <image> at beginning or end of the first user question
            image_placement = random.choice([0, 1])
            if image_placement == 0:
                convs.append({"from": "human", "value": f"<image>{q}"})
            else:
                convs.append({"from": "human", "value": f"{q}<image>"})
        else:
            convs.append({"from": "human", "value": q})

        convs.append({"from": "gpt", "value": a})

    data_dict = {
        "id": idx,
        "image": os.path.join("images", filename),
        "height": height,
        "width": width,
        "conversations": convs
    }
    return data_dict

def process_subset(snapshot_dir, subset_name, output_root):
    """
    Finds all train-*.parquet files for a given PixMo-Docs subset,
    loads them via load_dataset("parquet", ...),
    saves images, and writes a JSONL file in the 'pixmo_docs/' directory.
    """

    # 1) Gather all train parquet shards for this subset
    subset_glob = os.path.join(snapshot_dir, subset_name, "train-*.parquet")
    subset_files = sorted(glob.glob(subset_glob))
    if not subset_files:
        print(f"[{subset_name}] No train parquet files found at: {subset_glob}")
        return

    # 2) Load the dataset from these parquet shards
    #    Each file corresponds to a partition of the "train" subset
    ds = load_dataset("parquet", data_files=subset_files)["train"]
    print(f"[{subset_name}] Found {len(ds)} samples in {len(subset_files)} parquet shards.")

    # 3) Create images folder
    images_dir = os.path.join(output_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 4) Process and save images in parallel
    output_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        futures = {}
        for idx, entry in enumerate(ds):
            future = executor.submit(
                process_and_save_image,
                entry,
                subset_name,
                idx,
                images_dir
            )
            futures[future] = idx

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Processing {subset_name}",
        ):
            idx = futures[future]
            try:
                data_dict = future.result()
                output_data.append(data_dict)
            except Exception as e:
                print(f"Error on sample index {idx} for subset '{subset_name}': {e}")

    # 5) Sort output_data by ID (just in case) and write the final JSONL file
    output_data.sort(key=lambda x: x["id"])
    output_jsonl = os.path.join(output_root, f"{subset_name}_preprocessed.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[{subset_name}] Done! Saved {len(output_data)} samples.")
    print(f"[{subset_name}] Images saved to: {images_dir}")
    print(f"[{subset_name}] JSONL file: {output_jsonl}")

def main():

    download_dir = "./pixmo_docs_downloaded"

    # 1) Download only the train shards for each of the four subsets
    print("Downloading PixMo-Docs train splits from HuggingFace...")
    snapshot_dir = snapshot_download(
        repo_id="allenai/pixmo-docs",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=[
            "charts/train-*.parquet",
            "diagrams/train-*.parquet",
            "tables/train-*.parquet",
            "other/train-*.parquet"
        ]
    )
    print(f"Completed snapshot download to: {snapshot_dir}")

    # 2) Create the top-level output directory
    output_root = "pixmo_docs"
    os.makedirs(output_root, exist_ok=True)

    # 3) Process each of the four subsets
    for subset_name in SUBSETS:
        print(f"Processing subset '{subset_name}'...")
        process_subset(download_dir, subset_name, output_root)

    print("All processing complete!")

    shutil.rmtree(download_dir)

if __name__ == "__main__":
    main()


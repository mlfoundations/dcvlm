"""
Script to process the allenai/pixmo-points dataset.
- Downloads images from image_url using img2dataset.
- Generates question-answer pairs: "How many <label> are there in the image?" with answer as count.
- Handles label pluralization (e.g., "Book" -> "books").
- Avoids redundant image downloads by tracking URLs and reusing paths.
- Uses parallel processing for efficiency.
"""

import os
import json
import random
random.seed(0)  # for reproducibility
import pandas as pd
from huggingface_hub import snapshot_download
from img2dataset import download
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from urllib.parse import urlparse
import glob

# Output directory setup
output_dir = "./pixmo_points"
download_dir = "./pixmo_points_temp"
images_folder = os.path.join(output_dir, "images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)
os.makedirs(download_dir, exist_ok=True)

# Define question prompt variations
PROMPTS = [
    "How many {label} are there in the image?",
    "Can you tell me how many {label} are in this image?",
    "What is the number of {label} present in the image?",
    "How many {label} do you see in the image?",
    "Count the number of {label} in this image."
]

def process_label(label):
    """
    Process the label to lowercase and pluralize it appropriately.
    - Input: e.g., "Book"
    - Output: e.g., "books"
    """
    label = label.lower()
    # Simple pluralization rules (not exhaustive, but covers common cases)
    if label.endswith("y"):
        return label[:-1] + "ies"  # e.g., "category" -> "categories"
    elif label.endswith("ch") or label.endswith("sh"):
        return label + "es"  # e.g., "bus" -> "buses", "church" -> "churches"
    else:
        return label + "s"  # e.g., "book" -> "books"

def process_shard(shard_subdir):
    """Process a single shard folder to generate JSONL entries."""
    shard_path = os.path.join(images_folder, shard_subdir)
    shard_data = []

    all_json_files = [f for f in os.listdir(shard_path) if f.endswith(".json")]

    # Use shard number as base for unique IDs
    id_counter = int(shard_subdir) * 100_000

    for json_filename in tqdm(all_json_files, desc=f"Processing shard {shard_subdir}", leave=False):
        json_path = os.path.join(shard_path, json_filename)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("status") != "success":
            continue

        ext = meta.get("ext", "jpg")
        base_no_ext = os.path.splitext(json_filename)[0]
        image_filename = f"{base_no_ext}.{ext}"
        full_image_path = os.path.join(shard_subdir, image_filename)
        full_image_local = os.path.join(images_folder, full_image_path)
        if not os.path.exists(full_image_local):
            continue

        # Extract metadata
        width = meta.get("width") or meta.get("original_width")
        height = meta.get("height") or meta.get("original_height")
        url = meta.get("url")

        # Map URL to local image path (already downloaded by img2dataset)
        image_path = f"images/{full_image_path}"

        # Generate question and answer
        label = process_label(meta["label"])  # e.g., "Book" -> "books"
        count = meta["count"]
        question = random.choice(PROMPTS).format(label=label)
        answer = str(count)

        # Only keep the counting task samples
        if count is None:
            continue

        # Randomly place <image> at start or end to prevent prompt overfitting
        human_text = (
            f"<image>{question}" if random.choice([True, False])
            else f"{question}<image>"
        )

        convs = [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": answer}
        ]

        shard_data.append({
            "id": id_counter,
            "image": image_path,
            "height": int(height) if height else None,
            "width": int(width) if width else None,
            "conversations": convs,
            "original_url": url  # Include for tracking duplicates
        })
        id_counter += 1

    return shard_data

def deduplicate_and_finalize(data_list):
    """
    Deduplicate entries based on image URLs, keeping all questions but using a single image path.
    """
    url_to_path = {}
    final_data = []
    id_counter = 0

    for entry in data_list:
        url = entry.pop("original_url")  # Remove temporary field
        if url not in url_to_path:
            url_to_path[url] = entry["image"]
        entry["image"] = url_to_path[url]  # Use the first downloaded path for this URL
        entry["id"] = id_counter
        final_data.append(entry)
        id_counter += 1

    return final_data

def main():

    # Load dataset from Hugging Face
    snapshot_download(
        repo_id="allenai/pixmo-points",
        repo_type="dataset",
        allow_patterns=["data/train-*"],
        local_dir=download_dir)

    # Load all parquet files into a single DataFrame
    parquet_files = glob.glob(os.path.join(download_dir, "data/train-*.parquet"))
    df_list = [pd.read_parquet(file_) for file_ in parquet_files]
    df = pd.concat(df_list, ignore_index=True)

    # Keep only 'image_url' and 'caption' columns
    df = df[['image_url', 'points', 'count', 'label']].rename(columns={'image_url': 'url'})

    # Save combined DataFrame to Parquet file for img2dataset
    parquet_path = os.path.join(output_dir, "pixmo_points_tmp.parquet")
    df.to_parquet(parquet_path, index=False)

    # Download images using img2dataset
    print("Downloading images with img2dataset...")
    download(
        url_list=parquet_path,
        output_folder=images_folder,
        input_format="parquet",
        url_col="url",
        output_format="files",
	processes_count=16,
        thread_count=64,
        image_size=None,
        resize_mode="no",
        skip_reencode=True,
        number_sample_per_shard=100_000,
save_additional_columns=["count", "label", "points"]  # Add this to include count, points and label
    )

    # Process shards in parallel
    print("Starting parallel JSON parsing...")
    shard_folders = sorted(d for d in os.listdir(images_folder) if d.isdigit())

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_shard, shard_folders), total=len(shard_folders)))

    # Flatten results and deduplicate
    all_data = [item for sublist in results for item in sublist]
    final_data = deduplicate_and_finalize(all_data)

    # Write to JSONL
    out_jsonl = os.path.join(output_dir, "preprocessed.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Wrote {len(final_data)} entries to {out_jsonl}")

    # Clean up temporary files
    os.remove(parquet_path)
    shutil.rmtree(download_dir)

if __name__ == "__main__":
    main()
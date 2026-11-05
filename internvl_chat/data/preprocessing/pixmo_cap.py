import os
import json
import random
random.seed(0)  # for reproducibility
import pandas as pd
from huggingface_hub import snapshot_download
from img2dataset import download
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import glob

PROMPTS = [
    "Please provide a descriptive caption for this image.",
    "Write a concise and meaningful caption for the following image.",
    "Generate a short caption describing what you see in this image.",
    "Could you give me a succinct caption for this image?",
    "Please write a brief caption describing this image."
]

output_dir = "./pixmo_cap"
images_folder = os.path.join(output_dir, "images")

def process_shard(shard_subdir):
    shard_path = os.path.join(images_folder, shard_subdir)
    shard_data = []

    all_json_files = [f for f in os.listdir(shard_path) if f.endswith(".json")]

    id_counter = int(shard_subdir) * 10000

    for json_filename in tqdm(all_json_files, desc=f"shard {shard_subdir}", leave=False):
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

        caption = meta.get("caption", "No caption provided.")
        width = meta.get("width") or meta.get("original_width")
        height = meta.get("height") or meta.get("original_height")

        user_prompt = random.choice(PROMPTS)
        human_text = (
            f"<image>{user_prompt}" if random.choice([True, False])
            else f"{user_prompt}<image>"
        )

        convs = [
            {"from": "human", "value": human_text},
            {"from": "gpt",   "value": caption}
        ]

        shard_data.append({
            "id": id_counter,
            "image": f"images/{full_image_path}",
            "height": int(height) if height else None,
            "width": int(width) if width else None,
            "conversations": convs
        })
        id_counter += 1

    return shard_data

def main():

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download dataset
    snapshot_download(
        repo_id="allenai/pixmo-cap",
        repo_type="dataset",
        allow_patterns=["data/train-*"],
        local_dir=output_dir)

    # Load all parquet files into a single DataFrame
    parquet_files = glob.glob(os.path.join(output_dir, "data/train-*.parquet"))
    df_list = [pd.read_parquet(file_) for file_ in parquet_files]
    df = pd.concat(df_list, ignore_index=True)

    # Keep only 'image_url' and 'caption' columns
    df = df[['image_url', 'caption']].rename(columns={'image_url': 'url'})

    # Save combined DataFrame to Parquet file for img2dataset
    parquet_path = os.path.join(output_dir, "pixmo_cap_tmp.parquet")
    df.to_parquet(parquet_path, index=False)

    download(
        url_list=parquet_path,
        output_folder=images_folder,
        input_format="parquet",
        url_col="url",
        caption_col="caption",  # we keep the 'caption' so it ends up in the metadata
        output_format="files",  # "files" = each shard is a subfolder with images & metadata
        thread_count=64,        # you can adjust concurrency
	processes_count=16,
        image_size=None,        # if you want to skip resizing (we keep original size)
        resize_mode="no",       # skip resizing
        skip_reencode=True,     # skip re-encoding if the image is valid
        number_sample_per_shard=10_000,
        distributor="multiprocessing",  # or "pyspark" if you want to scale out
    )

    print("Starting parallel JSON parsing...")
    shard_folders = sorted(d for d in os.listdir(images_folder) if d.isdigit())

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_shard, shard_folders), total=len(shard_folders)))

    final_data = [item for sublist in results for item in sublist]

    out_jsonl = os.path.join(output_dir, "preprocessed.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Wrote {len(final_data)} entries to {out_jsonl}")

    # Clean up temporary files
    os.remove(parquet_path)
    shutil.rmtree(os.path.join(output_dir, "data"))

if __name__ == "__main__":
    main()
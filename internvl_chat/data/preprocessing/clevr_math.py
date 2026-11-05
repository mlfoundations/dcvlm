# Use hf_transfer for faster download
# https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads

from huggingface_hub import snapshot_download
from pathlib import Path
from tqdm import tqdm
import random
from PIL import Image
import json
import shutil
import zipfile
import concurrent.futures
import os

random.seed(0)


def extract_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        total_size = sum(zip_ref.getinfo(member).file_size for member in members)

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Extracting {zip_path.name}",
        ) as pbar:
            for member in members:
                zip_ref.extract(member, zip_path.parent / zip_path.stem)
                pbar.update(zip_ref.getinfo(member).file_size)

    zip_path.unlink()


download_dir = Path(__file__).parent / "clevr_math_download"
out_dir = Path(__file__).parent / "clevr_math"
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="dali-does/clevr-math",
    repo_type="dataset",
    allow_patterns=["data/*.zip"],
    local_dir=download_dir,
)

zip_files = list(download_dir.glob("data/*.zip"))
with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(os.cpu_count(), len(zip_files))
) as executor:
    executor.map(extract_zip, zip_files)

train_json_files = [
    download_dir / "data/clevr-math/clevr-math-train.json",
    download_dir / "data/clevr-math-multihop/clevr-math-train.json",
]
clevr_img_dir = download_dir / "data/CLEVR_v1.0/CLEVR_v1.0/images/train"

img_map = {}
img_counter = 0
entries = []
for i, train_json_file in enumerate(train_json_files):
    with open(train_json_file, "r") as f:
        data = json.load(f)

    for row in tqdm(
        data["questions"],
        ascii=True,
        desc=f"Processing file {i+1}/{len(train_json_files)}",
    ):
        img_path = clevr_img_dir / row["image_filename"]
        if img_path in img_map:
            img_name = img_map[img_path]
        else:
            with Image.open(img_path) as img:
                width, height = img.size
                img.verify()
            img_name = f"{img_counter}.png"
            img_map[img_path] = img_name
            img_path.rename(img_dir / img_name)
            img_counter += 1

        question, answer = row["question"], row["answer"]
        img_first = random.choice([0, 1])
        conversations = [
            {
                "from": "human",
                "value": (f"<image>{question}" if img_first else f"{question}<image>"),
            },
            {"from": "gpt", "value": answer},
        ]

        entry = {
            "id": len(entries),
            "image": f"images/{img_name}",
            "width": width,
            "height": height,
            "conversations": conversations,
        }
        entries.append(entry)

with open(out_dir / "preprocessed.jsonl", "w", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

shutil.rmtree(download_dir)

print("Preprocessing complete!")
print(f"Saved {len(entries)} samples to '{out_dir}/preprocessed.jsonl'")

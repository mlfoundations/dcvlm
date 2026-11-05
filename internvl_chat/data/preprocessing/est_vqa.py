from pathlib import Path
from tqdm import tqdm
import random
from PIL import Image
import shutil
import gdown
import json
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
                zip_ref.extract(member, download_dir)
                pbar.update(zip_ref.getinfo(member).file_size)

    zip_path.unlink()


download_dir = Path(__file__).parent / "est_vqa_download"
download_dir.mkdir(parents=True, exist_ok=True)

out_dir = Path(__file__).parent / "est_vqa"
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

ids = {
    "train.zip": "15q8Hqx3uQm8mx4w3PZtactJVe1BRXug5",
    "train.json": "1M3fWhFWltjM_nFR0ALynUIwXatwKXb26",
}

for name, id in ids.items():
    gdown.download(
        id=id,
        output=str(download_dir / name),
        fuzzy=True,
    )

zip_files = list(download_dir.glob("*.zip"))
max_workers = min(os.cpu_count(), len(zip_files))
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(extract_zip, zip_files)

json_file = download_dir / "train.json"
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

entries = []
img_counter = 0
for entry in tqdm(data, desc="Processing..."):
    img_path = download_dir / "train" / entry["image"]
    with Image.open(img_path) as img:
        width, height = img.size
        img.verify()
    new_name = f"{img_counter}.jpg"
    img_path.rename(img_dir / new_name)
    img_counter += 1

    for annotation in entry["annotation"]:
        question = annotation["question"]
        img_first = random.choice([0, 1])
        question = f"<image>{question}" if img_first else f"{question}<image>"
        conversations = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": annotation["answer"]},
        ]
        entries.append(
            {
                "id": len(entries),
                "image": f"images/{new_name}",
                "width": width,
                "height": height,
                "conversations": conversations,
            }
        )

with open(out_dir / "preprocessed.jsonl", "w", encoding="utf-8") as f:
    for entry in entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

shutil.rmtree(download_dir)

print("Preprocessing complete!")
print(f"Saved {len(entries)} samples to '{out_dir}/preprocessed.jsonl'")

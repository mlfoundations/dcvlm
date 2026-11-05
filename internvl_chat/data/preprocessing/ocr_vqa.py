# Use hf_transfer for faster download
# https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads

from huggingface_hub import snapshot_download
from pathlib import Path
from tqdm import tqdm
import random
from PIL import Image
import json
import shutil
import pandas as pd
from io import BytesIO

random.seed(0)

download_dir = Path(__file__).parent / "ocr_vqa_download"
out_dir = Path(__file__).parent / "ocr_vqa"
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="howard-hou/OCR-VQA",
    repo_type="dataset",
    allow_patterns=["data/train*.parquet"],
    local_dir=download_dir,
)

img_counter = 0
entries = []
parquet_files = list(download_dir.glob("data/*.parquet"))
for i, parquet_file in enumerate(parquet_files):
    df = pd.read_parquet(parquet_file)
    for row in tqdm(
        df.itertuples(),
        total=len(df),
        ascii=True,
        desc=f"Processing file {i+1}/{len(parquet_files)}...",
    ):
        image_bytes = BytesIO(row.image["bytes"])
        Image.open(image_bytes).verify()

        image_bytes.seek(0)
        with Image.open(image_bytes) as img:
            width, height = img.size
            img_name = f"sample_{img_counter}.png"
            img.save(img_dir / img_name)
        img_counter += 1

        questions = row.questions
        answers = row.answers
        for question, answer in zip(questions, answers):
            img_first = random.choice([0, 1])
            conversations = [
                {
                    "from": "human",
                    "value": (
                        f"<image>{question}" if img_first else f"{question}<image>"
                    ),
                },
                {"from": "gpt", "value": f"{answer}"},
            ]

            entry = {
                "id": len(entries),
                "image": f"images/{img_name}",
                "width": width,
                "height": height,
                "conversations": conversations,
            }
            entries.append(entry)

with open(out_dir / "preprocessed.jsonl", "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

shutil.rmtree(download_dir)

print("Preprocessing complete!")
print(f"Saved {len(entries)} samples to '{out_dir}/preprocessed.jsonl'")

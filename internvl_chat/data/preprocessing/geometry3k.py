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

download_dir = Path(__file__).parent / "geometry3k_download"
out_dir = Path(__file__).parent / "geometry3k"
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="lmms-lab/LLaVA-OneVision-Data",
    repo_type="dataset",
    allow_patterns=["Geometry3K(MathV360K)/*.parquet"],
    local_dir=download_dir,
)

entries = []
parquet_files = list(download_dir.glob("Geometry3K(MathV360K)/*.parquet"))
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
            img_name = f"sample_{len(entries)}.png"
            img.save(img_dir / img_name)

        assert len(row.conversations) == 2, "Expected two conversations per sample"
        question, answer = row.conversations
        question = question["value"]
        answer = answer["value"]

        _, _, question, *choices = question.split("\n")  # strip <image> and hint
        choices = "\n" + "\n".join(choices) if choices else ""
        question = question.replace("Question: ", "") + choices

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

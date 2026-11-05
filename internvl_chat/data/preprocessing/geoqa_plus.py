from huggingface_hub import snapshot_download
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
from io import BytesIO
import json
import shutil

output_dir = './geoqa_plus'
os.makedirs(output_dir, exist_ok=True)

img_dir = os.path.join(output_dir, "images")
os.makedirs(img_dir, exist_ok=True)

snapshot_download(
    repo_id="lmms-lab/LLaVA-OneVision-Data",
    repo_type="dataset",
    allow_patterns=["GeoQA+(MathV360K)/*.parquet"],
    local_dir=output_dir,
)

entries = []
parquet_files = os.listdir(os.path.join(output_dir, 'GeoQA+(MathV360K)'))

for i, parquet_file in enumerate(parquet_files):
    df = pd.read_parquet(os.path.join(output_dir, 'GeoQA+(MathV360K)', parquet_file))
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
            img.save(os.path.join(img_dir, img_name))
        conversations = list(row.conversations)

        entry = {
            "id": len(entries),
            "image": f"images/{img_name}",
            "width": width,
            "height": height,
            "conversations": conversations,
        }
        entries.append(entry)

with open(os.path.join(output_dir, "preprocessed.jsonl"), "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

shutil.rmtree(os.path.join(output_dir, 'GeoQA+(MathV360K)'))

# Use hf_transfer for faster download
# https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads

from huggingface_hub import snapshot_download
from pathlib import Path
from tqdm import tqdm
import random
from PIL import UnidentifiedImageError, Image
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
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


def extract_question_and_answer(json_file):
    """
    Extracts the question and answer from the json file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    if "en" in data:
        data = data["en"]
    if "question" in data and "CoT_reasoning" in data:
        question = data["question"]
        answer = data["CoT_reasoning"]
    elif "text" in data:
        text_list = data["text"]
        split_idx = text_list.index([])  # question is before the empty list
        question = " ".join(text_list[:split_idx])
        answer = " ".join(text_list[split_idx + 1 :])
    else:
        raise ValueError(f"Missing question or answer in {json_file}")
    question, answer = maybe_add_options(question, answer, data)
    return question, answer


def maybe_add_options(question, answer, data):
    if "choices" in data and "answer_index" in data:
        options = "Options:\n" + "\n".join(
            f"{chr(65 + i)}. {option}" for i, option in enumerate(data["choices"])
        )
        question = f"{question}\n{options}"
        final_answer = data["choices"][data["answer_index"]]
        answer = f"{answer}\nFinal answer: {final_answer}"
    return question, answer


download_dir = Path(__file__).parent / "mavis_function_download"
out_dir = Path(__file__).parent / "mavis_function"
out_dir.mkdir(parents=True, exist_ok=True)
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)


snapshot_download(
    repo_id="CaraJ/MAVIS-Function",
    repo_type="dataset",
    allow_patterns=["*.zip"],
    local_dir=download_dir,
)
print("Download complete!")


zip_files = list(download_dir.glob("*.zip"))
with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(zip_files))) as executor:
    executor.map(extract_zip, zip_files)

entries = []
img_counter = 0
skip_counter = 0
for subfolder in tqdm(
    list(download_dir.glob("*/*")), ascii=True, desc="Preprocessing..."
):
    json_file = subfolder / "qa.json"
    png_file = subfolder / "func1.png"

    if not json_file.exists() or not png_file.exists():
        skip_counter += 1
        continue
    try:
        Image.open(png_file).verify()
        with Image.open(png_file) as image:
            width, height = image.size
    except UnidentifiedImageError:
        skip_counter += 1
        continue

    img_name = f"sample_{img_counter}.png"
    png_file.rename(img_dir / img_name)
    img_counter += 1

    question, answer = extract_question_and_answer(json_file)
    img_first = random.choice([0, 1])
    question = f"<image>{question}" if img_first else f"{question}<image>"
    conversations = [
        {"from": "human", "value": question},
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

with open(out_dir / "preprocessed.jsonl", "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

shutil.rmtree(download_dir)

print("Preprocessing complete!")
print(f"Saved {len(entries)} samples to '{out_dir}/preprocessed.jsonl'")
print(f"Skipped {skip_counter} invalid samples")

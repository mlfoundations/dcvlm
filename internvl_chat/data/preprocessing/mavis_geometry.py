# Use hf_transfer for faster download
# https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads

from huggingface_hub import snapshot_download
from pathlib import Path
from tqdm import tqdm
import random
import json
import shutil
import zipfile
import concurrent.futures
import os
import cv2

random.seed(0)


def extract_qa(json_file, language, question_type):
    with open(json_file, "r") as f:
        data = json.load(f)

    question = data[f"text_{language}"][question_type]
    options = "Options:\n" + "\n".join(
        f"{chr(65 + i)}. {option}" for i, option in enumerate(data["choices"])
    )
    question = f"{question}\n{options}"

    final_answer = data["choices"][data["answer_index"]]
    xq = data[f"text_{language}"]
    answer = (
        f"{xq['CoT_reasoning']}\nFinal answer: {final_answer}"
    )

    return question, answer


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


def resize_and_save(source_path, dest_path, max_size=1024):
    """
    Process an image by resizing and saving it using OpenCV.

    Args:
        source_path (Path): Source path of the image
        dest_path (Path): Destination path to save the processed image
        max_size (int): Maximum size of the larger dimension

    Returns:
        tuple: (width, height) of the processed image

    Raises:
        FileNotFoundError: If source file doesn't exist
    """
    img = cv2.imread(str(source_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image at {source_path}")

    height, width = img.shape[:2]
    if width > height:
        new_width = min(width, max_size)
        new_height = int(height * (new_width / width))
    else:
        new_height = min(height, max_size)
        new_width = int(width * (new_height / height))

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest_path), img)

    return new_width, new_height


out_dir = Path(__file__).parent / "mavis_geometry"
out_dir.mkdir(parents=True, exist_ok=True)
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

download_dir = Path(__file__).parent / "mavis_geometry_download"
snapshot_download(
    repo_id="CaraJ/MAVIS-Geometry",
    repo_type="dataset",
    allow_patterns=["*.zip"],
    local_dir=download_dir,
)
print("Download complete!")


zip_files = list(download_dir.glob("*.zip"))
with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(os.cpu_count(), len(zip_files))
) as executor:
    executor.map(extract_zip, zip_files)


img_counter = 0
skip_counter = 0
entries = {"en": [], "zh": []}

for subdir in tqdm(
    list(download_dir.glob("RuleBaseGeo/*/*")),
    desc="Processing directories...",
    ascii=True,
):
    if not subdir.is_dir():
        continue

    base_name = int(subdir.name) + 1
    annotation_file = subdir / f"{base_name}_annotation.json"
    text_img = subdir / f"{base_name}_text.jpg"
    vision_img = subdir / f"{base_name}_vision_dominant.jpg"

    if not all(f.exists() for f in [annotation_file, text_img, vision_img]):
        skip_counter += 1
        continue

    try:
        txt_img_name = f"sample_{img_counter}_text.png"
        vis_img_name = f"sample_{img_counter}_vision.png"
        txt_img_size = resize_and_save(text_img, img_dir / txt_img_name)
        vis_img_size = resize_and_save(vision_img, img_dir / vis_img_name)
        img_counter += 1

        questions = [
            ("text_lite_question", txt_img_name, txt_img_size),
            ("text_dominant_question", txt_img_name, txt_img_size),
            ("vision_dominant_question", vis_img_name, vis_img_size),
        ]

        for language in ["en", "zh"]:
            for question_type, img_name, img_size in questions:
                width, height = img_size
                question, answer = extract_qa(annotation_file, language, question_type)
                img_first = random.choice([0, 1])
                question = f"<image>{question}" if img_first else f"{question}<image>"
                entry = {
                    "id": len(entries[language]),
                    "image": f"images/{img_name}",
                    "width": width,
                    "height": height,
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer},
                    ],
                }
                entries[language].append(entry)

    except (FileNotFoundError, json.JSONDecodeError):
        skip_counter += 1
        continue

for language in ["en", "zh"]:
    with open(out_dir / f"{language}_preprocessed.jsonl", "w", encoding="utf-8") as f:
        for entry in entries[language]:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

shutil.rmtree(download_dir)

print("Preprocessing complete!")
print(
    f"Saved {len(entries['en'])} English samples to '{out_dir}/en_preprocessed.jsonl'"
)
print(
    f"Saved {len(entries['zh'])} Chinese samples to '{out_dir}/zh_preprocessed.jsonl'"
)
print(f"Skipped {skip_counter} invalid samples")

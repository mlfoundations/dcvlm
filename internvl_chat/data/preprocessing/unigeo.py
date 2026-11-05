from pathlib import Path
from tqdm import tqdm
import random
from PIL import Image
import shutil
import pandas as pd
import gdown
import pickle
import json

random.seed(0)


def extract_qa(row, variant):
    extractors = {"proving": extract_qa_proving, "calculation": extract_qa_calculation}
    return extractors[variant](row)


def extract_qa_proving(row):
    replacements = {
        " ,": ",",
        " .": ".",
        " :": ":",
        "  ": " ",
        " )": ")",
        ",)": ")",
    }
    question = row.question
    for old, new in replacements.items():
        question = question.replace(old, new)
    assert len(row.statement) == len(
        row.reason
    ), "Statements and reasons have different lengths"
    answer = [
        f"Statement: {s}. Reason: {r}." for s, r in zip(row.statement, row.reason)
    ]
    answer = "\n".join(answer)
    return question, answer


def extract_qa_calculation(row):
    question = row.subject.rstrip("()") + ":"
    choices = [f"{chr(65 + i)}. {c}" for i, c in enumerate(row.choices)]
    choices = "\n".join(choices)
    question = f"{question}\n{choices}"
    return question, row.answer


entries = {"proving": [], "calculation": []}
download_dir = Path(__file__).parent / "unigeo_download"
out_dirs = {variant: Path(__file__).parent / f"unigeo_{variant}" for variant in entries}
img_dirs = {variant: out_dirs[variant] / "images" for variant in entries}

for img_dir in img_dirs.values():
    img_dir.mkdir(parents=True, exist_ok=True)

folder_url = "https://drive.google.com/drive/folders/1weZAzuxCyNsXK3NoUGo_7FWCAMnG4Mh4"
gdown.download_folder(folder_url, output=str(download_dir))
pickle_files = [p for p in download_dir.glob("*.pk") if p.stem != "sub_dataset_dict"]
if not pickle_files:
    raise FileNotFoundError("No pickle files found after download")

for i, pickle_file in enumerate(pickle_files):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
        data = pd.DataFrame(data)
        data.rename(columns={"img": "image"}, inplace=True)

    variant = pickle_file.stem.split("_")[0]
    assert variant in entries, f"Variant {variant} not in {entries.keys()}"

    for row in tqdm(
        data.itertuples(),
        total=len(data),
        ascii=True,
        desc=f"Processing file {i+1}/{len(pickle_files)}...",
    ):
        img = Image.fromarray(row.image.astype("uint8"))
        img.verify()
        width, height = img.size

        img_name = f"sample_{len(entries[variant])}.png"
        img.save(img_dirs[variant] / img_name)

        question, answer = extract_qa(row, variant)

        img_first = random.choice([0, 1])
        conversations = [
            {
                "from": "human",
                "value": (f"<image>{question}" if img_first else f"{question}<image>"),
            },
            {"from": "gpt", "value": answer},
        ]

        entry = {
            "id": len(entries[variant]),
            "image": f"images/{img_name}",
            "width": width,
            "height": height,
            "conversations": conversations,
        }
        entries[variant].append(entry)

for variant in entries:
    with open(out_dirs[variant] / "preprocessed.jsonl", "w", encoding="utf-8") as f:
        for entry in entries[variant]:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

shutil.rmtree(download_dir)

print("Preprocessing complete!")
for variant in entries:
    print(
        f"Saved {len(entries[variant])} samples to '{out_dirs[variant]}/preprocessed.jsonl'"
    )

from huggingface_hub import snapshot_download
import zipfile
import os
import pandas as pd
import json
import shutil

def json_to_jsonl(json_filepath, jsonl_filepath, images_dir):
    json_data = json.load(open(json_filepath))
    with open(jsonl_filepath, mode="w", encoding="utf-8") as jsonl_file:
        for id, item in enumerate(json_data):
            image_path = os.path.join(images_dir, item["img_name"])
            if os.path.exists(image_path):
                question = item["question"].strip()
                answer = item["answer"].strip()
                jsonl_data = {
                    "id": id,
                    "image": f"imgs/{item['img_name']}",
                    "conversations": [
                        {"from": "human", "value": f"<image>{question}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
                jsonl_file.write(json.dumps(jsonl_data) + "\n")
            else:
                print(f"Image not found: {image_path}")


def main():
    processed_dir = "./slake"

    print("Step 1: Downloading SLAKE dataset")
    snapshot_download(
        repo_id="BoKelvin/SLAKE", repo_type="dataset", local_dir=processed_dir
    )

    print("Step 2: Extracting SLAKE dataset")
    zip_files = ["imgs.zip", "KG.zip"]
    for zip_file in zip_files:
        with zipfile.ZipFile(os.path.join(processed_dir, zip_file), "r") as zip_ref:
            zip_ref.extractall(processed_dir)

    print("Step 3: Converting SLAKE dataset to jsonl format")
    json_filepath = f"{processed_dir}/train.json"
    jsonl_filepath = f"{processed_dir}/preprocessed.jsonl"
    images_dir = f"{processed_dir}/imgs"
    json_to_jsonl(json_filepath, jsonl_filepath, images_dir)

    shutil.rmtree(os.path.join(processed_dir, "KG"))
    shutil.rmtree(os.path.join(processed_dir, "__MACOSX"))

    os.remove(os.path.join(processed_dir, "KG.zip"))
    os.remove(os.path.join(processed_dir, "README.md"))
    os.remove(os.path.join(processed_dir, "imgs.zip"))
    os.remove(os.path.join(processed_dir, "train.json"))
    os.remove(os.path.join(processed_dir, "validation.json"))
    os.remove(os.path.join(processed_dir, "test.json"))
    os.remove(os.path.join(processed_dir, "mask.txt"))


if __name__ == "__main__":
    main()

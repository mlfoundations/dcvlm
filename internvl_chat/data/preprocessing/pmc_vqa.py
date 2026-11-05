from huggingface_hub import snapshot_download
import zipfile
import os
import pandas as pd
import json


def csv_to_jsonl(csv_filepath, jsonl_filepath, images_dir):
    df = pd.read_csv(csv_filepath)

    images_prefix = images_dir.split("/")[-1]
    answer_column = "Answer_label" if "Answer_label" in df.columns else "Answer"

    with open(jsonl_filepath, mode="w", encoding="utf-8") as jsonl_file:
        for id, row in df.iterrows():
            image_path = os.path.join(images_dir, row["Figure_path"])
            if os.path.exists(image_path):
                question_stem = row["Question"].strip()
                choices = [
                    f"A. {row['Choice A'].replace('A:', '').strip()}",
                    f"B. {row['Choice B'].replace('B:', '').strip()}",
                    f"C. {row['Choice C'].replace('C:', '').strip()}",
                    f"D. {row['Choice D'].replace('D:', '').strip()}",
                ]
                answer = row[answer_column].strip()
                question = (
                    f"<image>Question: {question_stem}\n\nOptions:\n"
                    + "\n".join(choices)
                    + "\n\nAnswer with the option's letter from the given choices directly."
                )
                jsonl_data = {
                    "id": id,
                    "image": f"{images_prefix}/{row['Figure_path']}",
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer},
                    ],
                }
                jsonl_file.write(json.dumps(jsonl_data) + "\n")
            else:
                print(f"Image not found: {image_path}")


def main():
    processed_dir = "./pmc_vqa"

    print("Step 1: Downloading PMC-VQA dataset")
    snapshot_download(
        repo_id="xmcmic/PMC-VQA", repo_type="dataset", local_dir=processed_dir
    )

    print("Step 2: Extracting PMC-VQA dataset")
    zip_files = ["images.zip", "images_2.zip"]
    for zip_file in zip_files:
        with zipfile.ZipFile(os.path.join(processed_dir, zip_file), "r") as zip_ref:
            zip_ref.extractall(processed_dir)

    print("Step 3: Converting PMC-VQA dataset to jsonl format")
    csv_filepath = f"{processed_dir}/train.csv"
    jsonl_filepath = f"{processed_dir}/preprocessed_0.jsonl"
    images_dir = f"{processed_dir}/images"
    csv_to_jsonl(csv_filepath, jsonl_filepath, images_dir)

    csv_filepath = f"{processed_dir}/train_2.csv"
    jsonl_filepath = f"{processed_dir}/preprocessed_1.jsonl"
    images_dir = f"{processed_dir}/figures"
    csv_to_jsonl(csv_filepath, jsonl_filepath, images_dir)

    os.remove(os.path.join(processed_dir, "README.md"))
    os.remove(os.path.join(processed_dir, "images.zip"))
    os.remove(os.path.join(processed_dir, "images_2.zip"))
    os.remove(os.path.join(processed_dir, "oa_comm_use_file_list.csv"))
    os.remove(os.path.join(processed_dir, "test.csv"))
    os.remove(os.path.join(processed_dir, "test_2.csv"))
    os.remove(os.path.join(processed_dir, "test_clean.csv"))
    os.remove(os.path.join(processed_dir, "train.csv"))
    os.remove(os.path.join(processed_dir, "train_2.csv"))

if __name__ == "__main__":
    main()

"""
SimChart9K dataset
https://drive.google.com/file/d/1M_NA3sIJNwCUfqB1HH0p4lsnEm3NbvsI/view
"""

import os
import random

import csv
import json
import gdown
import numpy as np
import pandas as pd
import shutil
import tqdm
import zipfile

def main():
    random.seed(0)
    np.random.seed(0)

    DOWNLOAD_LINK = "https://drive.google.com/file/d/1M_NA3sIJNwCUfqB1HH0p4lsnEm3NbvsI/view?usp=sharing"
    OUTPUT_DIR = "./simchart9k"
    TOTAL_VALID = 69964  # This marks the total length of valid preprocessed entries, since there are several entries that have broken formatting.

    def download_file_from_google_drive(url: str, destination: str):
        file_id = url.split('/')[5]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(download_url, destination, quiet=False)


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_file_from_google_drive(DOWNLOAD_LINK, f"{OUTPUT_DIR}/simchart9k.zip")

    zip_path = f"{OUTPUT_DIR}/simchart9k.zip"
    print(f"Extracting '{zip_path}' into '{OUTPUT_DIR}'...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)
    shutil.move(f"{OUTPUT_DIR}/SimChart9K/png", f"{OUTPUT_DIR}/images")
    shutil.move(f"{OUTPUT_DIR}/SimChart9K/table", f"{OUTPUT_DIR}/table")

    image_files = sorted(os.listdir(f"{OUTPUT_DIR}/images"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    table_files = sorted(os.listdir(f"{OUTPUT_DIR}/table"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    assert len(image_files) == len(table_files)


    ######################################
    preprocessed = []
    n_draws_per_image = 10
    uid = 0
    for image, table in tqdm.tqdm(zip(image_files, table_files), desc="Processing...", total=len(image_files)):
        table_path = os.path.join(OUTPUT_DIR, "table", table)

        # Custom situations:
        # 1. We have to read the header and the body separately, as sometimes the header can have multiple delimiters.
        # 2. Multiple values for y.
        try:
            x_vals, y_vals = [], []
            with open(table_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Read the first row (often the header)
                xy_names = next(reader)
                x_name = xy_names[0]
                y_names = xy_names[1:]
                for row in reader:
                    if len(row) - 1 > len(y_names):
                        raise ValueError(f"incorrect table formatting in {table}")
                    elif len(row) - 1 < len(y_names):
                        y_names = [", ".join(y_names)]
                    assert len(row)-1 == len(y_names), f"incorrect table formatting in {table}"
                    x_vals.append(row[0])

                    if not y_vals:
                        y_vals = [[] for _ in range(len(row[1:]))]
                    for i, val in enumerate(row[1:]):
                        y_vals[i].append(val)
        except:
            continue

        num_vals = len(x_vals)
        if num_vals <  n_draws_per_image:
            data_idcs = np.arange(num_vals)
        else:
            data_idcs = np.random.choice(np.arange(num_vals), n_draws_per_image, replace=False)
        for idx in data_idcs:
            x_val = x_vals[idx]

            y_idx = random.choice(list(range(len(y_names))))
            y_name = y_names[y_idx]
            y_val = y_vals[y_idx][idx]
        
            templated_q = random.choice([
                (f"When {x_name} is {x_val}, how high is the {y_name} according to the chart?", f"{y_val}"),
                (f"For the chart with x-axis '{x_name}' and y-axis '{y_name}', what does the chart indicate for {y_name} at {x_name} = {x_val}?", f"{y_val}"),
                (f"In the chart labeled '{x_name}' on the horizontal and '{y_name}' on the vertical, find the data point: x = {x_val}. What is the corresponding y value?", f"{y_val}"),
                (f"At {x_val} on the {x_name} axis, what is the value on the {y_name} axis?", f"{y_val}"),
                (f"When {y_name} is {y_val}, what does the chart indicate for {x_name}?", f"{x_val}"),
                (f"For the chart with x-axis '{x_name}' and y-axis '{y_name}', what is the {x_name} when {y_name} = {y_val}?", f"{x_val}"),
                (f"In the chart labeled '{x_name}' on the horizontal and '{y_name}' on the vertical, find the data point: y = {y_val}. What is the corresponding x value?", f"{x_val}"),
                (f"At {y_val} for {y_name}, what is the value on the {x_name} axis?", f"{x_val}"),
            ])

            human_q = f"<image>{templated_q[0]}" if random.choice([0, 1]) else f"{templated_q[0]}<image>"
            convs = [
                {"from": "human", "value": human_q},
                {"from": "gpt", "value": templated_q[1]},
            ]
            image_path = os.path.join("images", image)
            assert os.path.exists(os.path.join(OUTPUT_DIR, image_path))
            preprocessed.append({
                "id": uid,
                "image": image_path,
                "conversations": convs,
            })
            uid += 1

    assert len(preprocessed) == TOTAL_VALID, "Incorrect number of final preprocessed entries"

    with open(f"{OUTPUT_DIR}/preprocessed.jsonl", "w") as f:
        for entry in tqdm.tqdm(preprocessed, desc=f"Writing to jsonl-file"):
            f.write(json.dumps(entry) + "\n")

    os.remove(f"{OUTPUT_DIR}/simchart9k.zip")
    shutil.rmtree(f"{OUTPUT_DIR}/SimChart9K")
    shutil.rmtree(f"{OUTPUT_DIR}/table")

if __name__ == "__main__":
    main()
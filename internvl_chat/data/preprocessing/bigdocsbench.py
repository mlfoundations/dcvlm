from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

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
                zip_ref.extract(member, zip_path.parent / zip_path.stem)
                pbar.update(zip_ref.getinfo(member).file_size)

    zip_path.unlink()

download_dir = "./bigdocsbench_parquets"
os.makedirs(download_dir, exist_ok=True)
output_dir = "./bigdocsbench"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+"/images", exist_ok=True)

snapshot_download(
    repo_id="ServiceNow/BigDocs-Bench",
    repo_type="dataset",
    allow_patterns=["GUI-VQA/train*", "GUI2BBox/train*", "GUI2Summary/train*", "GUI2UserIntent/train*", "Image2Flow_(GraphViz)/train*", "Image2Flow_(JSON)/train*", "Table2LaTex/train*"],
    local_dir=download_dir
)

seeclick_url = "https://box.nju.edu.cn/f/6a804cf190dd490a808f/?dl=1"
seeclick_output = os.path.join(output_dir, "seeclick_data.zip")

# Download the file
subprocess.run(["wget", "-c", "-O", seeclick_output, seeclick_url])

# Extract the file
print("Extracting!")
extract_zip(Path(seeclick_output))

available_splits = [
    "GUI-VQA",
    "GUI2Summary",
    "Image2Flow_(GraphViz)",
    "Table2LaTex",
    "GUI2BBox",
    "GUI2UserIntent",
    "Image2Flow_(JSON)",
]

for split in available_splits:

    print("Processing split {}".format(split))

    new_jsonl_entries = []
    imgs_saved = 0
    parquets = [x for x in os.listdir(os.path.join(download_dir, split)) if x.endswith(".parquet")]
    for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
        df = pd.read_parquet(os.path.join(download_dir, split, parquet))
        for sample_index, row in tqdm(df.iterrows(), ascii=True):

            width, height = None, None

            if "img_id" in row:
                img_path_for_entry = os.path.join("seeclick_data", row["img_id"])
                assert os.path.exists(os.path.join(output_dir, img_path_for_entry)), "image path {} does not exist!!!".format(img_path_for_entry)
            elif "image" in row:
                assert "bytes" in row["image"], "No bytes in image!!!"
                assert len(row["image"]) == 2, "More than one image!!!"

                img_bytes = row["image"]["bytes"]  # Extract the byte content
                image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
                file_name = f"{output_dir}/images/{split}_parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
                width, height = image.size  # Get width and height
                image.save(file_name)  # Save the image
                imgs_saved += 1

                assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
                assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
                assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
            
                img_path_for_entry = "images/" + os.path.basename(file_name)

            # add entry to new jsonl list
            assert len(row["queries"]) == 1, "More than one human prompt!!!"
            assert len(row["annotations"]) == 1, "More than one gpt response!!!"

            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if image_beginning:
                if split in ["GUI2BBox"]:
                    convs = [{"from":"human", "value":"<image>{}".format(row["queries"][0])}, {"from":"gpt", "value":"<box>["+row["annotations"][0]+"]</box>"}]
                elif split in ["GUI2UserIntent"]:
                    convs = [{"from":"human", "value":"<image>{}".format(row["queries"][0].replace("[", "<box>[[").replace("]", "]]</box>"))}, {"from":"gpt", "value":row["annotations"][0]}]
                else:
                    convs = [{"from":"human", "value":"<image>{}".format(row["queries"][0])}, {"from":"gpt", "value":row["annotations"][0]}]
            else:
                if split in ["GUI2BBox"]:
                    convs = [{"from":"human", "value":"{}<image>".format(row["queries"][0])}, {"from":"gpt", "value":"<box>["+row["annotations"][0]+"]</box>"}]
                elif split in ["GUI2UserIntent"]:
                    convs = [{"from":"human", "value":"{}<image>".format(row["queries"][0].replace("[", "<box>[[").replace("]", "]]</box>"))}, {"from":"gpt", "value":row["annotations"][0]}]                    
                else:
                    convs = [{"from":"human", "value":"{}<image>".format(row["queries"][0])}, {"from":"gpt", "value":row["annotations"][0]}]

            if height is not None and width is not None:
                entry_ = {"id":len(new_jsonl_entries), "image": img_path_for_entry, "height": height, "width": width, "conversations": convs}
            else:
                entry_ = {"id":len(new_jsonl_entries), "image": img_path_for_entry, "conversations": convs}

            new_jsonl_entries.append(entry_)

    with open(os.path.join(output_dir, "{}_preprocessed.jsonl".format(split.lower())), 'w') as f:
        for entry in new_jsonl_entries:
            f.write(json.dumps(entry) + '\n')

    print("Preprocessing complete! Saved as 'bigdocsbench/{}_preprocessed.jsonl' with {} saved images and {} total samples.".format(split.lower(), imgs_saved, len(new_jsonl_entries)))

shutil.rmtree(download_dir)
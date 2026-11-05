###################################
# 1. download the spot-the-diff dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

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

snapshot_download(
    repo_id="HuggingFaceM4/the_cauldron",
    repo_type="dataset",
    allow_patterns=["spot_the_diff/*"],  # Only download files in the spot_the_diff folder
    local_dir="spot_the_diff_parquets"
)

output_dir = "./spot_the_diff/images"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
cnt = -1
imgs_saved = 0
parquets = [x for x in os.listdir("./spot_the_diff_parquets/spot_the_diff") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./spot_the_diff_parquets/spot_the_diff", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        images = row["images"]
        texts = row["texts"]
        # for spot_the_diff specifically we want to ensure we only have 2 images per sample, but can have multiple annotations
        assert len(images) == 2
        # save images
        file_names, heights, widths = [], [], []
        for i, img_dict in enumerate(images):
            img_bytes = img_dict["bytes"]  # Extract the byte content
            image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
            file_name = f"{output_dir}/parquet_{parquet_index}_sample_{sample_index}_{i}.png"  # Generate a unique name
            file_names.append(file_name)
            width, height = image.size  # Get width and height
            heights.append(height)
            widths.append(width)
            image.save(file_name)  # Save the image
            imgs_saved += 1
        assert len(file_names) == 2, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert len(heights) == 2, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert len(widths) == 2, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        # add entry to new jsonl list
        for t in texts:
            cnt += 1
            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if image_beginning:
                convs = [{"from":"human", "value":"<image><image>{}".format(t["user"])}, {"from":"gpt", "value":t["assistant"]}]
            else:
                convs = [{"from":"human", "value":"{}<image><image>".format(t["user"])}, {"from":"gpt", "value":t["assistant"]}]
            entry_ = {"id":cnt, "image": ['images/'+os.path.basename(f) for f in file_names], "height_list": heights, "width_list": widths, "conversations": convs}
            new_jsonl_entries.append(entry_)

with open('./spot_the_diff/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('spot_the_diff_parquets')

print("Preprocessing complete! Saved as 'spot_the_diff/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, cnt+1))

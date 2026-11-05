###################################
# 1. download the vqav2 dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
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
    allow_patterns=["vqav2/*"],  # Only download files in the vqav2 folder
    local_dir="vqav2_parquets"
)

output_dir = "./vqav2/images"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
cnt = -1
imgs_saved = 0
parquets = [x for x in os.listdir("./vqav2_parquets/vqav2") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./vqav2_parquets/vqav2", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        images = row["images"]
        texts = row["texts"]
        # for vqav2 specifically we want to ensure we only have 1 image per sample, but can have multiple annotations
        assert len(images) == 1
        # save image
        file_name, height, width = None, None, None
        for i, img_dict in enumerate(images):
            img_bytes = img_dict["bytes"]  # Extract the byte content
            image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
            file_name = f"{output_dir}/parquet_{parquet_index}_sample_{sample_index}_{i}.png"  # Generate a unique name
            width, height = image.size  # Get width and height
            image.save(file_name)  # Save the image
            imgs_saved += 1
        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        # add entry to new jsonl list
        for t in texts:
            cnt += 1
            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if image_beginning:
                convs = [{"from":"human", "value":"<image>{}".format(t["user"])}, {"from":"gpt", "value":t["assistant"]}]
            else:
                convs = [{"from":"human", "value":"{}<image>".format(t["user"])}, {"from":"gpt", "value":t["assistant"]}]
            entry_ = {"id":cnt, "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": convs}
            new_jsonl_entries.append(entry_)

with open('./vqav2/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('vqav2_parquets')

print("Preprocessing complete! Saved as 'vqav2/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, cnt+1))

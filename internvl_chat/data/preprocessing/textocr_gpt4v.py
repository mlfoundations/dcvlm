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

output_dir = './textocr_gpt4v'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

# first download all the images
# NOTE: we use the processed version of this dataset from LLaVA-OneVision-Data
# We tried to use the original dataset but the image mapping did not seem to match, hence we simply use the version from LLaVA-OneVision-Data
snapshot_download(
    repo_id="lmms-lab/LLaVA-OneVision-Data",
    repo_type="dataset",
    allow_patterns=["textocr(gpt4v)/*"],  # Only download files in the spot_the_diff folder
    local_dir='.'
)

# Define source and destination directories
source_dir = "textocr(gpt4v)"
destination_dir = "textocr_gpt4v"

# Move all files from source to destination
for file_name in os.listdir(source_dir):
    source_path = os.path.join(source_dir, file_name)
    destination_path = os.path.join(destination_dir, file_name)

    if os.path.isfile(source_path) and source_path.endswith('parquet'):  # Ensure it's a file, not a directory
        shutil.move(source_path, destination_path)

print(f"All parquet files moved from '{source_dir}' to '{destination_dir}'.")
shutil.rmtree(source_dir)

new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(output_dir) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True, total=len(parquets))):
    df = pd.read_parquet(os.path.join(output_dir, parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image = row["image"]
        convs = row["conversations"]
        image_id = row["id"]
        assert image is not None, 'image is None!!!'
        assert image['bytes'] is not None, 'image bytes are None!!!'
        assert image_id is not None, 'image-id is None!!!'
        assert convs is not None, 'conversations is None!!!'
        # save image
        img_bytes = image["bytes"]  # Extract the byte content
        image_ = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # Convert bytes to an image
        file_name = f"{output_dir}/images/parquet_{parquet_index}_sample_{sample_index}_id_{image_id}.png"  # Generate a unique name
        width, height = image_.size  # Get width and height
        image_.save(file_name)  # Save the image
        imgs_saved += 1

        assert height is not None, 'height is None!!!'
        assert width is not None, 'width is None!!!'

        convs = list(convs)
        assert len(convs) == 2, 'there is more than one turn in this sample!!!'

        entry_ = {"id":len(new_jsonl_entries), "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove(os.path.join(output_dir, 'train-00000-of-00006.parquet'))
os.remove(os.path.join(output_dir, 'train-00001-of-00006.parquet'))
os.remove(os.path.join(output_dir, 'train-00002-of-00006.parquet'))
os.remove(os.path.join(output_dir, 'train-00003-of-00006.parquet'))
os.remove(os.path.join(output_dir, 'train-00004-of-00006.parquet'))
os.remove(os.path.join(output_dir, 'train-00005-of-00006.parquet'))

print("Preprocessing complete! Saved as 'textocr_gpt4v/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))

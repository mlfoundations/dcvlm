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
    allow_patterns=["vistext/*"],  # Only download files in the vistext folder
    local_dir="vistext_parquets"
)

output_dir = "./vistext/images"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
cnt = -1
imgs_saved = 0
parquets = [x for x in os.listdir("./vistext_parquets/vistext") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./vistext_parquets/vistext", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        images = row["images"]
        texts = row["texts"]
        # for vistext specifically we want to ensure we only have 1 image per sample
        assert len(images) == 1
        # for vistext specifically we want to ensure we only have 1 annotation per sample
        assert len(texts) == 1
        # save image
        img_bytes = images[0]["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Convert bytes to an image
        file_name = f"{output_dir}/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1
        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        # add entry to new jsonl list
        for t in texts:
            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if image_beginning:
                convs = [{"from":"human", "value":"<image>{}".format(t["user"])}, {"from":"gpt", "value":t["assistant"]}]
            else:
                convs = [{"from":"human", "value":"{}<image>".format(t["user"])}, {"from":"gpt", "value":t["assistant"]}]
            entry_ = {"id":len(new_jsonl_entries), "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": convs}
            new_jsonl_entries.append(entry_)

with open('./vistext/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('vistext_parquets')

print("Preprocessing complete! Saved as 'vistext/preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))
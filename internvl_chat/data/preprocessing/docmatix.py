from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
import json
import shutil

random.seed(0)  # Ensure reproducibility

output_dir = "./docmatix"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/images', exist_ok=True)

snapshot_download(
    repo_id="HuggingFaceM4/Docmatix",
    repo_type="dataset",
    allow_patterns=["data/*parquet"],
    local_dir=os.path.join(output_dir, "docmatix_parquets")
)

imgs_saved = 0
parquets = sorted([x for x in os.listdir(os.path.join(output_dir, "docmatix_parquets", "data")) if x.endswith(".parquet")])
 
shard_threshold = 50  # we will have samples from 50 shards in each jsonl file i.e. roughly 1152 * 50 = 57,600 samples per jsonl 
entry_id = 0
shard_entries = []

for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True, total=len(parquets))):

    df = pd.read_parquet(os.path.join(output_dir, "docmatix_parquets", "data", parquet))

    for sample_index, row in tqdm(df.iterrows(), ascii=True, total=len(df)):
        images = row["images"]
        num_i = len(images)
        texts = row["texts"]

        # Save images
        file_names, heights, widths = [], [], []
        for i, img_dict in enumerate(images):
            img_bytes = img_dict["bytes"]
            image = Image.open(io.BytesIO(img_bytes))
            file_name = f"{output_dir}/images/parquet_{parquet_index}_sample_{sample_index}_{i}.png"
            file_names.append(file_name)
            width, height = image.size
            heights.append(height)
            widths.append(width)
            image.save(file_name)
            imgs_saved += 1
            del image
            del img_bytes
        
        del images

        # if no images are saved then discard this sample
        if len(file_names) == 0:
            continue

        assert file_names, f'No images found for parquet {parquet_index}, sample {sample_index}'
        assert heights, f'No image heights found for parquet {parquet_index}, sample {sample_index}'
        assert widths, f'No image widths found for parquet {parquet_index}, sample {sample_index}'

        # Create conversation entries
        convs = []
        for ind, t in enumerate(texts):
            if ind == 0:
                image_beginning = random.choice([0, 1])
                img_str = "<image>" * num_i
                if image_beginning:
                    convs.extend([{"from": "human", "value": f"{img_str}{t['user']}"},
                                  {"from": "gpt", "value": t["assistant"]}])
                else:
                    convs.extend([{"from": "human", "value": f"{t['user']}{img_str}"},
                                  {"from": "gpt", "value": t["assistant"]}])
            else:
                convs.extend([{"from": "human", "value": t["user"]},
                              {"from": "gpt", "value": t["assistant"]}])

        assert len(convs) == len(texts) * 2, "Mismatch in conversation length!"

        del texts

        # Add entry to the current shard buffer
        entry_ = {
            "id": entry_id,
            "image": ['images/' + os.path.basename(f) for f in file_names],
            "height_list": heights,
            "width_list": widths,
            "conversations": convs
        }
        shard_entries.append(entry_)
        entry_id += 1

    del df  # Free up memory

    # Check if we need to write this shard
    if (parquet_index + 1) % shard_threshold == 0:

        write_index = parquet_index // shard_threshold

        shard_filename = os.path.join(output_dir, f"preprocessed_{write_index}.jsonl")
        
        # Write entries for this range
        with open(shard_filename, 'w') as f:
            for entry in shard_entries:
                f.write(json.dumps(entry) + '\n')

        print(f"Saved {len(shard_entries)} samples to {shard_filename}")
        shard_entries = []  # Clear buffer

# Write final set of shards
if len(shard_entries) > 0:

    write_index = parquet_index // shard_threshold

    shard_filename = os.path.join(output_dir, f"preprocessed_{write_index}.jsonl")
    
    # Write entries for this range
    with open(shard_filename, 'w') as f:
        for entry in shard_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(shard_entries)} samples to {shard_filename}")

# Remove the downloaded parquet files to free up space
shutil.rmtree(os.path.join(output_dir, "docmatix_parquets"))

print(f"Preprocessing complete! Saved samples across shards in {output_dir} with {imgs_saved} images saved and {write_index + 1} total shards.")

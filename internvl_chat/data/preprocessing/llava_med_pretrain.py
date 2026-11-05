from huggingface_hub import snapshot_download
import json
import os
import shutil
from PIL import Image
import io
import shutil
import random
random.seed(0)
from tqdm import tqdm
import pandas as pd

def main(chunk_id, processed_dir):

    snapshot_download(
        repo_id=f"Shubhangi29/llava_med_alignment_500k_chunk_{chunk_id}", repo_type="dataset", local_dir=processed_dir+"/chunk_{}".format(chunk_id)
    )

    new_jsonl_entries = []
    imgs_saved = 0
    pdir = os.path.join(processed_dir, "chunk_{}".format(chunk_id))
    parquets = [x for x in os.listdir(os.path.join(pdir, "data")) if x.endswith(".parquet") and "train" in x]
    for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
        df = pd.read_parquet(os.path.join(pdir, "data", parquet))
        for sample_index, row in tqdm(df.iterrows(), ascii=True):
            image_dict = row["image"]
            conversations = row["conversations"]

            # save image
            file_name, height, width = None, None, None
            img_bytes = image_dict["bytes"]  # Extract the byte content
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Convert bytes to an image
            file_name = f"{processed_dir}/images/chunk_{chunk_id}_parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
            width, height = image.size  # Get width and height
            image.save(file_name)  # Save the image
            imgs_saved += 1
            assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
            assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
            assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
            # add entry to new jsonl list
            entry_ = {"id":len(new_jsonl_entries), "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": list(conversations)}
            new_jsonl_entries.append(entry_)

    with open(os.path.join(processed_dir, 'preprocessed_{}.jsonl'.format(chunk_id)), 'w') as f:
        for entry in new_jsonl_entries:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":

    processed_dir = "./llava_med_pretrain"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(processed_dir+"/images", exist_ok=True)

    main(1, processed_dir)
    main(2, processed_dir)
    main(3, processed_dir)
    main(4, processed_dir)
    main(5, processed_dir)

    shutil.rmtree(os.path.join(processed_dir, "chunk_1"))
    shutil.rmtree(os.path.join(processed_dir, "chunk_2"))
    shutil.rmtree(os.path.join(processed_dir, "chunk_3"))
    shutil.rmtree(os.path.join(processed_dir, "chunk_4"))
    shutil.rmtree(os.path.join(processed_dir, "chunk_5"))

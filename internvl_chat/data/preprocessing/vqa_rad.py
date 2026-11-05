import json
from huggingface_hub import snapshot_download
import os
import shutil
from PIL import Image
from tqdm import tqdm
import pandas as pd
import io
import random
random.seed(0)

def main():

    processed_dir = "./vqa_rad"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(processed_dir+"/images", exist_ok=True)

    snapshot_download(
        repo_id="flaviagiammarino/vqa-rad", repo_type="dataset", local_dir=processed_dir
    )

    new_jsonl_entries = []
    imgs_saved = 0
    parquets = [x for x in os.listdir(os.path.join(processed_dir, "data")) if x.endswith(".parquet") and "train" in x]
    for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
        df = pd.read_parquet(os.path.join(processed_dir, "data", parquet))
        for sample_index, row in tqdm(df.iterrows(), ascii=True):
            image_dict = row["image"]
            question = row["question"]
            answer = row["answer"]
            # save image
            file_name, height, width = None, None, None
            img_bytes = image_dict["bytes"]  # Extract the byte content
            image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
            file_name = f"{processed_dir}/images/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
            width, height = image.size  # Get width and height
            image.save(file_name)  # Save the image
            imgs_saved += 1
            assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
            assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
            assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
            # add entry to new jsonl list
            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if image_beginning:
                convs = [{"from":"human", "value":"<image>{}".format(question)}, {"from":"gpt", "value":answer}]
            else:
                convs = [{"from":"human", "value":"{}<image>".format(question)}, {"from":"gpt", "value":answer}]
            entry_ = {"id":len(new_jsonl_entries), "image": 'images/'+os.path.basename(file_name), "height": height, "width": width, "conversations": convs}
            new_jsonl_entries.append(entry_)

    with open(os.path.join(processed_dir, 'preprocessed.jsonl'), 'w') as f:
        for entry in new_jsonl_entries:
            f.write(json.dumps(entry) + '\n')


    shutil.rmtree(os.path.join(processed_dir, "data"))
    shutil.rmtree(os.path.join(processed_dir, "scripts"))
    os.remove(os.path.join(processed_dir, "README.md"))

if __name__ == "__main__":
    main()

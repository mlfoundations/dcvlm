###################################
# 1. download the sea-vl dataset using the snapshot_download below
# 2. run this script from the same folder to get the preprocessed version!
# 3. estimated time to run: 50 hours --> resume option for when you are on a cluster
# with limited running time
###################################

import numpy as np
from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import re
import shutil

# I know it's not the best solution, but since it requires too many gpu/hours I need to run multiple jobs. Os.listdir and other solution may not be deterministic, so ugly but correct >> nice but wrong.
PARQUETS = ['train-00199-of-00219.parquet', 'train-00109-of-00219.parquet', 'train-00162-of-00219.parquet', 'train-00027-of-00028.parquet', 'train-00151-of-00219.parquet', 'train-00034-of-00219.parquet', 'train-00184-of-00219.parquet', 'train-00021-of-00028.parquet', 'train-00038-of-00219.parquet', 'train-00095-of-00219.parquet', 'train-00187-of-00219.parquet', 'train-00172-of-00219.parquet', 'train-00074-of-00219.parquet', 'train-00156-of-00219.parquet', 'train-00010-of-00028.parquet', 'train-00072-of-00219.parquet', 'train-00129-of-00219.parquet', 'train-00051-of-00219.parquet', 'train-00186-of-00219.parquet', 'train-00102-of-00219.parquet', 'train-00025-of-00028.parquet', 'train-00152-of-00219.parquet', 'train-00123-of-00219.parquet', 'train-00118-of-00219.parquet', 'train-00193-of-00219.parquet', 'train-00213-of-00219.parquet', 'train-00104-of-00219.parquet', 'train-00153-of-00219.parquet', 'train-00188-of-00219.parquet', 'train-00033-of-00219.parquet', 'train-00189-of-00219.parquet', 'train-00101-of-00219.parquet', 'train-00198-of-00219.parquet', 'train-00001-of-00028.parquet', 'train-00014-of-00219.parquet', 'train-00200-of-00219.parquet', 'train-00208-of-00219.parquet', 'train-00135-of-00219.parquet', 'train-00086-of-00219.parquet', 'train-00214-of-00219.parquet', 'train-00020-of-00028.parquet', 'train-00027-of-00219.parquet', 'train-00030-of-00219.parquet', 'train-00116-of-00219.parquet', 'train-00158-of-00219.parquet', 'train-00040-of-00219.parquet', 'train-00039-of-00219.parquet', 'train-00185-of-00219.parquet', 'train-00091-of-00219.parquet', 'train-00100-of-00219.parquet', 'train-00019-of-00028.parquet', 'train-00082-of-00219.parquet', 'train-00076-of-00219.parquet', 'train-00009-of-00219.parquet', 'train-00071-of-00219.parquet', 'train-00119-of-00219.parquet', 'train-00054-of-00219.parquet', 'train-00042-of-00219.parquet', 'train-00122-of-00219.parquet', 'train-00166-of-00219.parquet', 'train-00016-of-00219.parquet', 'train-00005-of-00028.parquet', 'train-00150-of-00219.parquet', 'train-00017-of-00028.parquet', 'train-00006-of-00219.parquet', 'train-00194-of-00219.parquet', 'train-00049-of-00219.parquet', 'train-00080-of-00219.parquet', 'train-00066-of-00219.parquet', 'train-00174-of-00219.parquet', 'train-00036-of-00219.parquet', 'train-00138-of-00219.parquet', 'train-00007-of-00219.parquet', 'train-00177-of-00219.parquet', 'train-00022-of-00219.parquet', 'train-00149-of-00219.parquet', 'train-00003-of-00219.parquet', 'train-00065-of-00219.parquet', 'train-00063-of-00219.parquet', 'train-00191-of-00219.parquet', 'train-00011-of-00028.parquet', 'train-00026-of-00219.parquet', 'train-00182-of-00219.parquet', 'train-00017-of-00219.parquet', 'train-00018-of-00219.parquet', 'train-00190-of-00219.parquet', 'train-00019-of-00219.parquet', 'train-00026-of-00028.parquet', 'train-00139-of-00219.parquet', 'train-00113-of-00219.parquet', 'train-00000-of-00219.parquet', 'train-00048-of-00219.parquet', 'train-00000-of-00028.parquet', 'train-00007-of-00028.parquet', 'train-00218-of-00219.parquet', 'train-00060-of-00219.parquet', 'train-00053-of-00219.parquet', 'train-00169-of-00219.parquet', 'train-00078-of-00219.parquet', 'train-00197-of-00219.parquet', 'train-00031-of-00219.parquet', 'train-00206-of-00219.parquet', 'train-00085-of-00219.parquet', 'train-00073-of-00219.parquet', 'train-00006-of-00028.parquet', 'train-00126-of-00219.parquet', 'train-00203-of-00219.parquet', 'train-00140-of-00219.parquet', 'train-00134-of-00219.parquet', 'train-00105-of-00219.parquet', 'train-00146-of-00219.parquet', 'train-00207-of-00219.parquet', 'train-00005-of-00219.parquet', 'train-00057-of-00219.parquet', 'train-00210-of-00219.parquet', 'train-00202-of-00219.parquet', 'train-00157-of-00219.parquet', 'train-00096-of-00219.parquet', 'train-00097-of-00219.parquet', 'train-00115-of-00219.parquet', 'train-00205-of-00219.parquet', 'train-00163-of-00219.parquet', 'train-00075-of-00219.parquet', 'train-00068-of-00219.parquet', 'train-00001-of-00219.parquet', 'train-00092-of-00219.parquet', 'train-00170-of-00219.parquet', 'train-00098-of-00219.parquet', 'train-00008-of-00028.parquet', 'train-00144-of-00219.parquet', 'train-00103-of-00219.parquet', 'train-00130-of-00219.parquet', 'train-00108-of-00219.parquet', 'train-00018-of-00028.parquet', 'train-00215-of-00219.parquet', 'train-00003-of-00028.parquet', 'train-00012-of-00028.parquet', 'train-00147-of-00219.parquet', 'train-00021-of-00219.parquet', 'train-00164-of-00219.parquet', 'train-00136-of-00219.parquet', 'train-00099-of-00219.parquet', 'train-00204-of-00219.parquet', 'train-00009-of-00028.parquet', 'train-00141-of-00219.parquet', 'train-00159-of-00219.parquet', 'train-00020-of-00219.parquet', 'train-00015-of-00219.parquet', 'train-00002-of-00219.parquet', 'train-00084-of-00219.parquet', 'train-00037-of-00219.parquet', 'train-00055-of-00219.parquet', 'train-00077-of-00219.parquet', 'train-00165-of-00219.parquet', 'train-00044-of-00219.parquet', 'train-00161-of-00219.parquet', 'train-00058-of-00219.parquet', 'train-00132-of-00219.parquet', 'train-00069-of-00219.parquet', 'train-00029-of-00219.parquet', 'train-00023-of-00028.parquet', 'train-00004-of-00028.parquet', 'train-00089-of-00219.parquet', 'train-00046-of-00219.parquet', 'train-00023-of-00219.parquet', 'train-00050-of-00219.parquet', 'train-00176-of-00219.parquet', 'train-00090-of-00219.parquet', 'train-00110-of-00219.parquet', 'train-00137-of-00219.parquet', 'train-00012-of-00219.parquet', 'train-00056-of-00219.parquet', 'train-00028-of-00219.parquet', 'train-00016-of-00028.parquet', 'train-00181-of-00219.parquet', 'train-00142-of-00219.parquet', 'train-00067-of-00219.parquet', 'train-00117-of-00219.parquet', 'train-00083-of-00219.parquet', 'train-00107-of-00219.parquet', 'train-00201-of-00219.parquet', 'train-00180-of-00219.parquet', 'train-00010-of-00219.parquet', 'train-00052-of-00219.parquet', 'train-00032-of-00219.parquet', 'train-00183-of-00219.parquet', 'train-00059-of-00219.parquet', 'train-00154-of-00219.parquet', 'train-00013-of-00219.parquet', 'train-00043-of-00219.parquet', 'train-00041-of-00219.parquet', 'train-00212-of-00219.parquet', 'train-00167-of-00219.parquet', 'train-00168-of-00219.parquet', 'train-00022-of-00028.parquet', 'train-00087-of-00219.parquet', 'train-00173-of-00219.parquet', 'train-00002-of-00028.parquet', 'train-00133-of-00219.parquet', 'train-00008-of-00219.parquet', 'train-00160-of-00219.parquet', 'train-00148-of-00219.parquet', 'train-00155-of-00219.parquet', 'train-00128-of-00219.parquet', 'train-00209-of-00219.parquet', 'train-00111-of-00219.parquet', 'train-00045-of-00219.parquet', 'train-00047-of-00219.parquet', 'train-00195-of-00219.parquet', 'train-00106-of-00219.parquet', 'train-00178-of-00219.parquet', 'train-00093-of-00219.parquet', 'train-00070-of-00219.parquet', 'train-00062-of-00219.parquet', 'train-00120-of-00219.parquet', 'train-00025-of-00219.parquet', 'train-00011-of-00219.parquet', 'train-00216-of-00219.parquet', 'train-00015-of-00028.parquet', 'train-00112-of-00219.parquet', 'train-00127-of-00219.parquet', 'train-00145-of-00219.parquet', 'train-00121-of-00219.parquet', 'train-00211-of-00219.parquet', 'train-00143-of-00219.parquet', 'train-00004-of-00219.parquet', 'train-00064-of-00219.parquet', 'train-00079-of-00219.parquet', 'train-00088-of-00219.parquet', 'train-00179-of-00219.parquet', 'train-00024-of-00028.parquet', 'train-00217-of-00219.parquet', 'train-00171-of-00219.parquet', 'train-00125-of-00219.parquet', 'train-00094-of-00219.parquet', 'train-00014-of-00028.parquet', 'train-00114-of-00219.parquet', 'train-00196-of-00219.parquet', 'train-00124-of-00219.parquet', 'train-00081-of-00219.parquet', 'train-00024-of-00219.parquet', 'train-00192-of-00219.parquet', 'train-00035-of-00219.parquet', 'train-00013-of-00028.parquet', 'train-00131-of-00219.parquet', 'train-00061-of-00219.parquet', 'train-00175-of-00219.parquet']
# Get filename and remove extension
DS_NAME = os.path.basename(__file__).split(".")[0] 
# Temporary this for download
TEMP_DIR = f"./{DS_NAME}_temp"
# Final directory for the images
IMAGES_DIR = f"./{DS_NAME}/images"
# Final path for annotations
ANNOTATIONS_FILE = lambda idx: f"./{DS_NAME}/preprocessed_{idx}.jsonl"

# Prompt to be used for the caption dataset
# taken from: https://arxiv.org/pdf/2304.08485 tab. 11
LLAVA_DESC_SHORT_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo’s key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]


def main(_range):

    # Parquets split
    from_idx = int(_range.split(":")[0])
    to_idx = int(_range.split(":")[1])

    print(f"Processing sea-vl from {from_idx} to {to_idx}.")
    
    snapshot_download(
        repo_id="SEACrowd/sea-vl_crowdsourced",
        repo_type="dataset",
        allow_patterns=["data/*"],
        local_dir=TEMP_DIR
    )

    snapshot_download(
        repo_id="SEACrowd/sea-vl_crawling",
        repo_type="dataset",
        allow_patterns=["data/*"],
        local_dir=TEMP_DIR
    )


    os.makedirs(IMAGES_DIR, exist_ok=True)


    for parquet_idx in range(from_idx, to_idx):
        new_jsonl_annotation = []

        print("Processing {} [{} of {}]".format(PARQUETS[parquet_idx], parquet_idx+1-from_idx, to_idx-from_idx), flush=True)

        df = pd.read_parquet(f"{TEMP_DIR}/data/{PARQUETS[parquet_idx]}")

        for sample_idx, row in df.iterrows():

            # sea-vl has only one image/caption per sample
            img_dict = row["image"]
            caption = row["caption"]
            
            # sea-vl crawling save captions as numpy arrays
            # this is incredibly strange
            if isinstance(caption, np.ndarray):
                caption = ". ".join(caption)

            prompt = random.choice(LLAVA_DESC_SHORT_PROMPTS)

            # save image
            file_name, height, width = None, None, None
            
            img_bytes = img_dict["bytes"]  # Extract the byte content
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Convert bytes to an image
            file_name = f"{IMAGES_DIR}/parquet_{parquet_idx}_sample_{sample_idx}.png"  # Generate a unique name
            width, height = image.size  # Get width and height
            width = int(width)
            height = int(height)

            image.save(file_name)  # Save the image
            assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_idx, sample_idx)
            assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_idx, sample_idx)
            assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_idx, sample_idx)

            
            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if image_beginning:
                conv = [{"from":"human", "value":"<image>{}".format(prompt)}, {"from":"gpt", "value": caption}]
            else:
                conv = [{"from":"human", "value":"{}<image>".format(prompt)}, {"from":"gpt", "value": caption}]

            entry_ = {"id": sample_idx, "image": 'images/' + os.path.basename(file_name), "height": height, "width": width, "conversations": conv}
            new_jsonl_annotation.append(entry_)
            
        with open(ANNOTATIONS_FILE(parquet_idx), 'w') as f:
            for entry in new_jsonl_annotation:
                f.write(json.dumps(entry) + '\n')
                
        print(
            f"Preprocessing complete! Saved as '{ANNOTATIONS_FILE(parquet_idx)}' with {sample_idx} saved images and {sample_idx} total samples.",
            flush=True
        )
        # Close resource
        del df
        # Remove file
        os.remove(f"{TEMP_DIR}/data/{PARQUETS[parquet_idx]}")

    if len(os.listdir(f"{TEMP_DIR}/data/")) == 0:
        print("Processing Done ...")
        shutil.rmtree(TEMP_DIR)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--range", type=str, default="0:247") 
    args = parser.parse_args()

    main(args.range)

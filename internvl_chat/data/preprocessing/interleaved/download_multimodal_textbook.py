###################################
# 1. download the multimodal_textbook dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

# path where files are temporarily stored
DOWNLOAD_DIR = "multimodal_textbook_parquets"

def main():
    snapshot_download(
        repo_id="DAMO-NLP-SG/multimodal_textbook",
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR,
    )
    os.system("cat multimodal_textbook_parquets/dataset_images_interval_7.tar.gz.part_* > multimodal_textbook_parquets/dataset_images_interval_7.tar.gz")
    shutil.unpack_archive(
        "multimodal_textbook_parquets/dataset_images_interval_7.tar.gz",
        DOWNLOAD_DIR
    )

if __name__ == "__main__":
    main()

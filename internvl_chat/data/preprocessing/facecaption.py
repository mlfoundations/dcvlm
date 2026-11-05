import os
from tqdm import tqdm
import pandas as pd
from huggingface_hub import snapshot_download
from img2dataset import download

if __name__ == "__main__":
    parquets_dir = "facecaption_parquets"
    download_dir = "facecaption_download"
    facecaption_preprocessed_path = os.path.join(parquets_dir, "facecaption_preprocessed.parquet")
    
    if not os.path.exists(facecaption_preprocessed_path):
        print("Pre-processing the parquet file...")
        snapshot_download(
            repo_id="OpenFace-CQUPT/FaceCaption-15M",
            repo_type="dataset",
            allow_patterns=["FaceCaption-v2.parquet"],  # Only download train file
            local_dir=parquets_dir
        )
        df = pd.read_parquet(os.path.join(parquets_dir, "FaceCaption-v2.parquet"))
        rows = []
        for sample_index, row in tqdm(df.iterrows(), ascii=True, total=len(df)):
            url = row["url"].split(",")[0].strip()
            caption = row["caption"][0]
            image_id = row["image_id"]
            rows.append({"url": url, "image_id": "image_id", "caption": caption})
        # Write ready-to-use parquet file for img2dataset
        pd.DataFrame(rows).to_parquet(facecaption_preprocessed_path)
    assert os.path.exists(facecaption_preprocessed_path)
    print("Starting the download...")
    download(
        processes_count=16,
        thread_count=64,
        url_list=facecaption_preprocessed_path,
        disable_all_reencoding=True,
        output_folder=download_dir,
        output_format="webdataset",
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=True,
        number_sample_per_shard=10_000,
        save_additional_columns=["image_id"],
    )
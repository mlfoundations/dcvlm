"""
ProVision-10M dataset
https://huggingface.co/datasets/Salesforce/ProVision-10M
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import random
import requests
import shutil
import time

import argparse
from img2dataset import download
import json
import pandas as pd
import tqdm
import zipfile


def main(args):
    random.seed(0)

    output_dir = "./provision10m"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    BASE_URL = "https://huggingface.co/datasets/Salesforce/ProVision-10M/resolve/main/data/"
    parquets = [
        "dcm_mc_2_img-00000-of-00002.parquet",
        "dcm_mc_2_img-00001-of-00002.parquet",
        "dcm_mc_3_img-00000-of-00003.parquet",
        "dcm_mc_3_img-00001-of-00003.parquet",
        "dcm_mc_3_img-00002-of-00003.parquet",
        "dcm_mc_4_img-00000-of-00004.parquet",
        "dcm_mc_4_img-00001-of-00004.parquet",
        "dcm_mc_4_img-00002-of-00004.parquet",
        "dcm_mc_4_img-00003-of-00004.parquet",
        "dcm_sa_2_img-00000-of-00002.parquet",
        "dcm_sa_2_img-00001-of-00002.parquet",
        "dcm_sa_3_img-00000-of-00003.parquet",
        "dcm_sa_3_img-00001-of-00003.parquet",
        "dcm_sa_3_img-00002-of-00003.parquet",
        "dcm_sa_4_img-00000-of-00003.parquet",
        "dcm_sa_4_img-00001-of-00003.parquet",
        "dcm_sa_4_img-00002-of-00003.parquet",
        "dcs_mc-00000-of-00003.parquet",
        "dcs_mc-00001-of-00003.parquet",
        "dcs_mc-00002-of-00003.parquet",
        "dcs_sa-00000-of-00003.parquet",
        "dcs_sa-00001-of-00003.parquet",
        "dcs_sa-00002-of-00003.parquet",
        "vgm_mc_2_img-00000-of-00002.parquet",
        "vgm_mc_2_img-00001-of-00002.parquet",
        "vgm_mc_3_img-00000-of-00002.parquet",
        "vgm_mc_3_img-00001-of-00002.parquet",
        "vgm_mc_4_img-00000-of-00002.parquet",
        "vgm_mc_4_img-00001-of-00002.parquet",
        "vgm_sa_2_img-00000-of-00002.parquet",
        "vgm_sa_2_img-00001-of-00002.parquet",
        "vgm_sa_3_img-00000-of-00002.parquet",
        "vgm_sa_3_img-00001-of-00002.parquet",
        "vgm_sa_4_img-00000-of-00002.parquet",
        "vgm_sa_4_img-00001-of-00002.parquet",
        "vgs_mc-00000-of-00002.parquet",
        "vgs_mc-00001-of-00002.parquet",
        "vgs_sa-00000-of-00002.parquet",
        "vgs_sa-00001-of-00002.parquet",
    ]

    def download_file(url, output_path):
        """
        A helper function to download a single file from a URL into output_path.
        Raises an exception if download fails, which the caller can catch.
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


    ######## first download and unzip files
    iterator = tqdm.tqdm(parquets, desc="Downloading parquets...")
    for parquet in iterator:
        train_parquet_path = os.path.join(output_dir, parquet)
        iterator.set_description(f"Downloading {parquet}")
        download_file(os.path.join(BASE_URL, parquet), train_parquet_path)


    ##############################################################################
    # 1) Gather all datacomp download links.
    ##############################################################################
    download_jobs = []
    preprocessed = []
    download_count = 0
    unique_ids = set()

    print("Getting DataComp Download Paths...")
    start = time.time()
    target_parquets = parquets[:23]
    for i_p, parquet_file in enumerate(target_parquets):
        parquet_path = os.path.join(output_dir, parquet_file)
        df = pd.read_parquet(parquet_path)

        iterator = tqdm.tqdm(df.iterrows(), desc=f"Collecting entries for parquet {i_p+1}/{len(target_parquets)}...", total=len(df), leave=False)
        for _, row in iterator:
            img_download_paths = row["data_path"]
            if isinstance(img_download_paths, str):
                img_download_paths = [img_download_paths]
            else:
                img_download_paths = img_download_paths.tolist()

            question = row["question"]
            answer   = row["answer"]

            # Build the list of download tasks
            for img_download_path in img_download_paths:
                uid = img_download_path.split(":")[0]
                download_path = ":".join(img_download_path.split(":")[1:])

                # Only download if we haven't already encountered this uid
                if uid not in unique_ids:
                    unique_ids.add(uid)
                    download_jobs.append(download_path)
                    download_count += 1
                    iterator.set_postfix_str(f"{download_count} urls to download.")
    print("Collecting download paths took {}s".format(time.time() - start))
    df = pd.DataFrame(download_jobs, columns=['url'])
    df.to_csv(os.path.join(output_dir, 'dc_urls_to_download.csv'), index=False)


    ##############################################################################
    # 2) Downloading datacomp images.
    ##############################################################################
    # Img2Dataset parallel download.
    download_start = time.time()
    download(
        processes_count=8,
        thread_count=16,
        url_list=os.path.join(output_dir, "dc_urls_to_download.csv"),
        resize_mode="no",
        output_folder=os.path.join(output_dir, "images"),
        output_format="files",
        input_format="csv",
        url_col="url",
    )
    print("Full image download completed in {}s.".format(time.time() - download_start))

    #### Create a DC folder
    os.makedirs(os.path.join(output_dir, "images", "dc_images"), exist_ok=True)
    # Extract valid download paths:
    folders = sorted(os.listdir(os.path.join(output_dir, "images")))
    folders = [x for x in folders if x[:2] == "00" and "." not in x]
    successful_downloads = {}
    for i, folder in enumerate(folders):
        folder_path = os.path.join(output_dir, "images", folder)
        files = os.listdir(folder_path)
        json_files = sorted([os.path.join(folder_path, x) for x in files if ".json" in x])
        for json_file in tqdm.tqdm(json_files, desc=f"Checking {i+1}/{len(folders)}", leave=False):
            with open(json_file, 'r') as file:
                status_file = json.load(file)
            success = status_file["status"] == "success"
            url = status_file["url"]
            if success:
                successful_downloads[url] = json_file.replace(".json", ".jpg")
    print(f"Successfully downloaded {len(successful_downloads)} datacomp files.")


    ##############################################################################
    # 3) Creating datacomp preprocessed list.
    ##############################################################################

    # Move images to dc folder, and place into processed-list.
    print("Creating datacomp preprocessed file...")
    start = time.time()
    id_count = 0
    retrieved = set()
    target_parquets = parquets[:23]
    for i_p, parquet_file in enumerate(target_parquets):
        parquet_path = os.path.join(output_dir, parquet_file)
        df = pd.read_parquet(parquet_path)

        iterator = tqdm.tqdm(df.iterrows(), desc=f"Collecting entries for parquet {i_p+1}/{len(target_parquets)}...", total=len(df), leave=False)
        for row_i, row in iterator:
            img_download_paths = row["data_path"]
            if isinstance(img_download_paths, str):
                img_download_paths = [img_download_paths]
            else:
                img_download_paths = img_download_paths.tolist()

            question = row["question"]
            answer   = row["answer"]
            # Build the list of download tasks
            valid_download = True
            uids = []
            retrieve_paths = []
            for img_download_path in img_download_paths:
                uid = img_download_path.split(":")[0]
                download_path = ":".join(img_download_path.split(":")[1:])
                valid_download = download_path in successful_downloads
                if not valid_download:
                    break
                uids.append(uid)
                retrieve_paths.append(successful_downloads[download_path])

            if not valid_download:
                continue
            else:
                image_paths = []
                for uid, retrieve_path in zip(uids, retrieve_paths):
                    image_path = f"images/dc_images/{uid}.jpg"
                    if uid not in retrieved:
                        retrieved.add(uid)
                        shutil.copy(retrieve_path, os.path.join(output_dir, image_path))
                    image_paths.append(image_path)

                human_q = question
                if "<image>" not in question:
                    image_str = " ".join(["<image>" for _ in range(len(image_paths))])
                    human_q = f"{image_str}{question}" if random.choice([0, 1]) else f"{question}{image_str}"
                convs = [
                    {"from": "human", "value": human_q},
                    {"from": "gpt", "value": answer},
                ]
                preprocessed.append({
                    "id": id_count,
                    "image": image_paths,
                    "conversations": convs
                })
                id_count += 1
    print("Created datacomp preprocessed file in {}s".format(time.time() - start))


    ##############################################################################
    # 4) Downloading and preprocessing optional visual genome.
    ##############################################################################
    if args.download_vg:
        #### Download Visual Genome Image Data
        url1 = "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
        url2 = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
        path1, path2 = "images.zip", "images2.zip"
        folder1, folder2 = "VG_100K", "VG_100K_2"
        vg_path = os.path.join(output_dir, "images", "vg")
        os.makedirs(vg_path, exist_ok=True)
        for i, (url, path, folder) in enumerate(zip([url1, url2], [path1, path2], [folder1, folder2])):
            print(f"Downloading Visual Genome ({i+1}/2)...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(os.path.join(vg_path, path), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        ### Extract VG
        for i, (path, folder) in enumerate(zip([path1, path2], [folder1, folder2])):
            folder_path = os.path.join(vg_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Extracting Visual Genome ({i+1}/2)...")
            with zipfile.ZipFile(os.path.join(vg_path, path), 'r') as zip_ref:
                zip_ref.extractall(vg_path)

        ### Check correct download counts.
        successful_files = set()
        for i, (path, folder) in enumerate(zip([path1, path2], [folder1, folder2])):
            files = os.listdir(os.path.join(vg_path, folder))
            for file in tqdm.tqdm(files, leave=False):
                successful_files.add(f"vg/{folder}/{file}")
        assert len(successful_files) == 108249, f"Incorrect file count for VG files! Have {len(successful_files)}, need 108249!"

        ### Create preprocessed file.
        print("Creating visual genome preprocessed file...")
        start = time.time()
        target_parquets = parquets[23:]
        preprocessed_vg = []
        for i_p, parquet_file in enumerate(target_parquets):
            parquet_path = os.path.join(output_dir, parquet_file)
            df = pd.read_parquet(parquet_path)

            iterator = tqdm.tqdm(df.iterrows(), desc=f"Collecting entries for parquet {i_p+1}/{len(target_parquets)}...", total=len(df), leave=False)
            for row_i, row in iterator:
                image_paths = row["data_path"]
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                else:
                    image_paths = image_paths.tolist()

                question = row["question"]
                answer   = row["answer"]

                human_q = question
                if "<image>" not in question:
                    image_str = " ".join(["<image>" for _ in range(len(image_paths))])
                    human_q = f"{image_str}{question}" if random.choice([0, 1]) else f"{question}{image_str}"
                convs = [
                    {"from": "human", "value": human_q},
                    {"from": "gpt", "value": answer},
                ]
                preprocessed_vg.append({
                    "id": id_count,
                    "image": [f"images/{x}" for x in image_paths],
                    "conversations": convs
                })
                id_count += 1

        print("Created visual genome preprocessed file in {}s".format(time.time() - start))
        preprocessed = preprocessed + preprocessed_vg

        #### Remove unnecessary zip files.
        for path in [path1, path2]:
            os.remove(os.path.join(vg_path, path))


    ##############################################################################
    # 5) Dump final preprocessed file.
    ##############################################################################
    TOTAL_VALID = 20_305_788
    if len(preprocessed) != TOTAL_VALID:
        print(f"Incorrect number of final preprocessed entries, likely due to broken links. Have: {len(preprocessed)}. Target: {TOTAL_VALID}.")

    CHUNK_SIZE = 1_000_000
    n_chunks = len(preprocessed) // 1_000_000 + 1
    for i in range(n_chunks):        
        with open(f"{output_dir}/preprocessed_{i}.jsonl", "w") as f:
            for entry in tqdm.tqdm(preprocessed[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE], desc=f"Writing to jsonl-file ({i+1}/{n_chunks})"):
                f.write(json.dumps(entry) + "\n")


    ##############################################################################
    # 6) Remove final redundancies.
    ##############################################################################

    folders = sorted(os.listdir(os.path.join(output_dir, "images")))
    folders = [x for x in folders if x not in ["dc_images", "vg"]]
    for folder in tqdm.tqdm(folders, desc="Deleting redundant downloads..."):
        path = os.path.join(output_dir, "images", folder)
        if "." in folder:
            os.remove(path)
        else:
            shutil.rmtree(path)

    folders = sorted(os.listdir(os.path.join(output_dir)))
    folders = [x for x in folders if "preprocessed" not in x and x not in ["images"]]
    for folder in tqdm.tqdm(folders, desc="Deleting redundant parquets and files..."):
        os.remove(os.path.join(output_dir, folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_vg', action="store_true")
    args = parser.parse_args()
    main(args)
import wget
import os
from natsort import natsorted
import json
from huggingface_hub import snapshot_download
import tarfile
from collections import defaultdict as ddict
import gzip
import subprocess
from tqdm import tqdm

### copied from: https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/blob/main/extract.py
def extract_tar(tar_file, destination_folder):
    print(f"Extracting {tar_file}...")
    subprocess.run(["tar", "-xzf", tar_file, "-C", destination_folder])
    print(f"Extraction complete for {tar_file}.")
    os.remove(tar_file)
    print(f"Removed tar file {tar_file} to save space.")

output_dir = './cambrian_gpt4o'
os.makedirs(output_dir, exist_ok=True)
wget.download('https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/jsons/gpt4o_60k.jsonl', out=output_dir)

to_download = []
for line in open(os.path.join(output_dir, 'gpt4o_60k.jsonl'), 'r'):
    image_folder = json.loads(line)['image'].split('/')[0]
    if image_folder not in to_download:
        to_download.append(image_folder)

print("List of datasets to download", to_download)
snapshot_download(repo_id='nyu-visionx/Cambrian-10M', repo_type="dataset", allow_patterns=[f'{dataset_name}.tar.gz*' for dataset_name in to_download], local_dir=output_dir)

for dataset_name in to_download:

    if os.path.isdir(os.path.join(output_dir, dataset_name)):
        print("{} already exists skipping!".format(dataset_name))
        continue

    print("Processing dataset {}".format(dataset_name))

    if dataset_name == "dvqa":
        all_tars = ['dvqa.tar.gz']
    elif dataset_name == "textvqa":
        all_tars = ['textvqa.tar.gz']
    else:
        all_tars = [f for f in os.listdir(output_dir) if f'{dataset_name}.tar.gz' in f]

    assert len(all_tars) >= 1

    if dataset_name in ['allava', 'data_engine']:

        # merge and then extract
        output_tar = os.path.join(output_dir, f"{dataset_name}.tar.gz")

        if not os.path.exists(output_tar):
            print("Merging the tars first!")
            parts = natsorted([os.path.join(output_dir, x) for x in all_tars if x.startswith(dataset_name) and x[len(dataset_name+".tar.gz_part"):].isdigit()])

            # Merge the tar parts into a single tar.gz file
            with open(output_tar, "wb") as output_file:
                for part in parts:
                    with open(part, "rb") as part_file:
                        output_file.write(part_file.read())
                    print(f"Merged {part} into {output_tar}")

            # Delete part files to save space
            for part in parts:
                os.remove(part)
                print(f"Removed part file {part}")

        else:

            print("Merged tar already exists!")

        extract_tar(output_tar, output_dir)

    else:

        # directly extract
        extract_tar(os.path.join(output_dir, all_tars[0]), output_dir)

output_data = []
input_path = os.path.join(output_dir, 'gpt4o_60k.jsonl')
skipped_samples = 0

with open(input_path, 'r') as f:
    for line in tqdm(f, desc="Processing entries"):
        entry = json.loads(line)
        img_path = entry['image']
        dataset_name = img_path.split('/')[0]

        new_img_path = img_path
        if not os.path.exists(os.path.join(output_dir, new_img_path)):
            print("Path does not exist, new datasets: {}: {}".format(dataset_name, new_img_path))
            skipped_samples += 1
            continue

        entry['image'] = new_img_path
        entry['id'] = int(entry['id'])
        output_data.append(entry)

print("Skipped {} samples due to missing or corrupted images!".format(skipped_samples))

with open(f'{output_dir}/preprocessed.jsonl', 'w') as f:
    for entry in output_data:
        f.write(json.dumps(entry) + '\n')

os.remove(os.path.join(output_dir, "gpt4o_60k.jsonl"))
for f in os.listdir(output_dir):
    if '.tar.gz' in f:
        print("Removing {}".format(f))
        os.remove(os.path.join(output_dir, f))

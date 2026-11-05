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
import zipfile

snapshot_download(
    repo_id="laolao77/MMDU",
    repo_type="dataset",
    allow_patterns=["mmdu-45k*"],  # Only download files in the spot_the_diff folder
    local_dir="."
)

output_dir = "./mmdu"
os.makedirs(output_dir, exist_ok=True)

# move files to correct folder
source_dir = "."
target_folder = "mmdu"

# Files to move
files_to_move = ["mmdu-45k.json", "mmdu-45k_pics.zip"]

for file_ in files_to_move:
    src_path = os.path.join(source_dir, file_)
    dest_path = os.path.join(target_folder, file_)
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"Moved {file_} to {target_folder}")
    else:
        print(f"File {file_} not found")

# Unzip inside target folder
zip_dest_path = os.path.join(output_dir, files_to_move[1])
with zipfile.ZipFile(zip_dest_path, 'r') as zip_ref:
    zip_ref.extractall(target_folder)
    print(f"Extracted zip into {target_folder}")

# Delete the zip file
os.remove(zip_dest_path)

# Rename folder of images
os.rename(os.path.join(output_dir, "mmdu-45k_pics"), os.path.join(output_dir, "images"))

## Reformat the json
new_jsonl_entries = []

with open(os.path.join(output_dir, "mmdu-45k.json")) as f:
    json_contents = json.load(f)

for index, content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    image_list = ['images/'+os.path.basename(x) for x in content["image"]]

    for img in image_list:
        assert os.path.exists(os.path.join(output_dir, img)), '{} does not exist!!!'.format(img)

    convs = content["conversations"]

    formatted_convs = []
    tot_imgs_orig = 0
    tot_imgs_reformat = 0
    for c_ in convs:
        d_ = {}
        if c_["from"] == "user":
            d_["from"] = "human"
            # there are some cases where <image> appears as a standard str but we cant have that since in our case its a token
            # hence we first remove all <image> strs, convert them to simple "image" and then replace the actual image tokens
            d_["value"] = c_["value"].replace("<image>", "image").replace("<ImageHere>", "<image>")
            assert c_["value"].count("<ImageHere>") == d_["value"].count("<image>"), "Counts of image tokens don't match!!!"
        elif c_["from"] == "assistant":
            d_["from"] = "gpt"
            d_["value"] = c_["value"]
            # there are some cases when the <image> token appears in the gpt output, for their dataset its fine since they dont use <image> as a token
            # but for us its critical to replace it, since we cannot have <image> tokens lying around in the gpt outputs, else the model will learn to output it
            if "<image>" in d_["value"]:
                print("Replacing <image> token in gpt output!")
                d_["value"] = d_["value"].replace("<image>", "image")
        else:
            raise ValueError("Unknown key! Should be in [assistant, user]")
        formatted_convs.append(d_)

        # intentionally doing it more inefficiently to have a double sanity check!
        tot_imgs_orig += c_["value"].count("<ImageHere>")
        tot_imgs_reformat += d_["value"].count("<image>")

    assert tot_imgs_orig == tot_imgs_reformat, 'Image counts dont match b/w new and old json contents!!!'
    assert tot_imgs_orig == len(image_list), 'Image counts dont match b/w image list and json contents!!!'
    assert len(formatted_convs) == len(convs), 'New convs and old convs arent of same length!!!'

    entry_ = {"id":len(new_jsonl_entries), "image": image_list, "conversations": formatted_convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove(os.path.join(output_dir, "mmdu-45k.json"))

print("Preprocessing complete! Saved as 'mmdu/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
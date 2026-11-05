import gdown
import os
import io
from PIL import Image
from tqdm import tqdm
import random
import json
import shutil
import tarfile
import sys
import numpy as np
import cv2 as cv

random.seed(0)  # Ensure reproducibility

output_dir = "./visualmrc"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/images', exist_ok=True)

# Google Drive file ID extracted from the URL: https://drive.google.com/file/d/1kxCIG6Nns7L79i5s7xmvKS92foIDRm4T/view
file_id = "1kxCIG6Nns7L79i5s7xmvKS92foIDRm4T"
file_url = f"https://drive.google.com/uc?id={file_id}"

# Output file name
output_filename = os.path.join(output_dir, "visualmrc.zip")

# Download the file
print('Downloading...')
gdown.download(file_url, output_filename, quiet=False)

# Extract the tar.gz file
print('Extracting folder...')
with tarfile.open(output_filename, 'r:gz') as tar:
    tar.extractall(output_dir)
    print(f"Extracted files to {output_dir}")

# Optionally remove the tar file after extraction
os.remove(output_filename)

annotation_jsonl_file_path = os.path.join(output_dir, "VisualMRC_official", "annotation.jsonl")
image_location_jsonl_file_path = os.path.join(output_dir, "VisualMRC_official", "image_location.jsonl")

###### Next, run a conversion script from screenshots to the images specified by bboxes

with open(image_location_jsonl_file_path) as fp:
    lines = fp.readlines()

for line in tqdm(enumerate(lines), ascii=True, total=len(lines)):
    line = line[1]
    data = json.loads(line.strip())
    screenshot_im_path = os.path.join(output_dir, "VisualMRC_official", data['screenshot_filename'])
    image_im_path = os.path.join(output_dir, data['image_filename'])
    
    screenshot_im = Image.open(screenshot_im_path)
    loc = data['image_loc']
    image_pil = screenshot_im.convert(mode='RGB').crop((loc["x"], loc["y"], loc["x"]+loc["width"], loc["y"]+loc["height"]))

    dirname = os.path.dirname(image_im_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    image_pil.save('{}'.format(image_im_path))

###### Now, run a conversion script to preprocess the jsonl file

new_jsonl_entries = []

with open(annotation_jsonl_file_path) as fp:
    json_contents = [json.loads(x) for x in fp.readlines()]

for content in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
    content = content[1]
    image = content["image_filename"]
    qa = content["qa_data"]

    assert os.path.exists(os.path.join(output_dir, image)), "{} does not exist!!!".format(image)

    convs = []
    for index, t in enumerate(qa):
        if index == 0:
            image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
            if image_beginning:
                convs.extend([{"from":"human", "value":"<image>{}".format(t["question"]["text"])}, {"from":"gpt", "value":t["answer"]["text"]}])
            else:
                convs.extend([{"from":"human", "value":"{}<image>".format(t["question"]["text"])}, {"from":"gpt", "value":t["answer"]["text"]}])
        else:
            convs.extend([{"from":"human", "value":"{}".format(t["question"]["text"])}, {"from":"gpt", "value":t["answer"]["text"]}])

    entry_ = {"id":len(new_jsonl_entries), "image": image, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "VisualMRC_official"))

print("Preprocessing complete! Saved as 'visualmrc/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

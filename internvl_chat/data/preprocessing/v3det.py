import os
from tqdm import tqdm
import shutil
import json
import zipfile
import requests
import random
random.seed(0)

V3DET_TRAIN_ANNOTS = 'https://huggingface.co/datasets/yhcao/V3Det_Backup/resolve/main/v3det_2023_v1_train.json'
V3DET_TRAIN_URLS = []
for i in range(0, 20):
    V3DET_TRAIN_URLS.append('https://huggingface.co/datasets/yhcao/V3Det_Backup/resolve/main/zips/'+f"{i:05}"+'.zip')

ROOT_DIR = './v3det'
IMAGE_DIR = './v3det/images'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")


# Function to unzip files
def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")


# Function to convert bbox format from xywh to xyxy
# COCO by default uses x_min, y_min, w, h format
# InternVL 2.5 format requires x_min, y_min, x_max, y_max format
def convert_bbox_xywh_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]


# Function to normalize bbox coordinates, taken from InternVL documentations
def normalize_coordinates(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    normalized_bbox = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_bbox


# Download existing train annotations
train_json_path = os.path.join(ROOT_DIR, 'train.json')
download_file(V3DET_TRAIN_ANNOTS, train_json_path)

# Download and unzip all of the associated zip files...
for idx, url in enumerate(V3DET_TRAIN_URLS):
    dpath_ = os.path.join(ROOT_DIR, str(idx)+'.zip')
    if not os.path.exists(dpath_):
        print("Downloading {}".format(dpath_))
        download_file(url, dpath_)
        unzip_file(dpath_, IMAGE_DIR)

###### Read and create preprocessed jsonl file

train_json_annotations = json.load(open(train_json_path))

img_id_to_annot = dict()
for img in train_json_annotations['images']:
    img_id_to_annot[img['id']] = {'height': img['height'], 'width': img['width'], 'file_name': img['file_name'], 'annotations': dict()}

cat_id_to_name = dict()
for cat in train_json_annotations['categories']:
    cat_id_to_name[cat['id']] = cat['name']

for annot in train_json_annotations['annotations']:
    cat_name = cat_id_to_name[annot['category_id']]
    
    if cat_name not in img_id_to_annot[annot['image_id']]['annotations'].keys():
        img_id_to_annot[annot['image_id']]['annotations'][cat_name] = []
    
    img_id_to_annot[annot['image_id']]['annotations'][cat_name].append(annot)


new_jsonl_entries = []

missing_entries = []

# add entry to new jsonl list
for img_id, img_info in tqdm(img_id_to_annot.items(), ascii=True, total=len(img_id_to_annot.keys())):
    image_path = img_info['file_name']
    if not os.path.exists(os.path.join(ROOT_DIR, image_path)):
        missing_entries.append(os.path.join(ROOT_DIR, image_path))
        continue
    img_width, img_height = img_info['width'], img_info['height']

    prompt_ = "Please detect and label all objects in the following image and mark their positions."
    response_ = "Sure, I will detect and label all objects in the image and mark their positions.\n\n"

    # For detection, each response contains all of the class names and associated bboxes together, separated by \n
    for cat_name, annotations in img_info['annotations'].items():
        cat_bboxes = []
        for annot in annotations:
            bbox = convert_bbox_xywh_xyxy(annot['bbox'])
            bbox = normalize_coordinates(bbox, img_width, img_height)
            cat_bboxes.append(bbox)

        det_response_begin_ = "<ref>{}</ref><box>[".format(cat_name)
        for idx, bbox in enumerate(cat_bboxes):
            det_response_begin_ += str(bbox)

            if idx < len(cat_bboxes)-1:
                det_response_begin_ += ", "

        det_response_ = det_response_begin_ + "]</box>\n"
        response_ += det_response_

    image_beginning = random.choice([0, 1])
    if image_beginning:
        convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    
    entry_ = {"id": len(new_jsonl_entries), "image": image_path, "conversations": convs, "height": img_height, "width": img_width}
    new_jsonl_entries.append(entry_)

with open(os.path.join(ROOT_DIR, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'v3det/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

# Given that the official V3Det downloading script was broken, a Huggingface back-up repository was utilized to download the images
# It misses around 7K images out of 183354 images
print("In addition, there were {} missing entries in the downloaded images...".format(len(missing_entries)))
os.remove(train_json_path)

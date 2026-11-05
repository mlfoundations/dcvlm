import json
import os
import sys
import requests
from tqdm import tqdm
import pickle
import shutil
import zipfile
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

REFCOCO_ZIP = "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
REFCOCOG_ZIP = "https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"
REFCOCOPLUS_ZIP = "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"
output_dir = './refcoco'

######## All utility functions

# Function to get coco images path
def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2014", f"{image_id:012}")

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


def download_file(url, output_file):
    """Downloads a file from the given URL with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_file, "wb") as file, tqdm(
        desc=output_file,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

    print(f"Download complete: {output_file}")

# Function to collate all of the relevant annotations from coco-style and pickled annotation files
def get_full_annotations(coco_style_annots, pickle_annots):
    img_id_to_annot = dict()
    for img in coco_style_annots['images']:
        img_id_to_annot[img['id']] = {'height': img['height'], 'width': img['width'], 'file_name': img['file_name'], 'annotations': []}

    coco_style_info = dict()
    for annot in coco_style_annots['annotations']:
        coco_style_info[annot['id']] = dict()
        coco_style_info[annot['id']]['bbox'] = annot['bbox']
        
    for annot in pickle_annots:
        new_annot = dict()

        # only collect the annotations belonging to the train splits for any of the RefCOCO datasets
        if annot['split'] != 'train':
            continue
        
        #if 'annotations' not in img_id_to_annot[annot['image_id']].keys():
        #    img_id_to_annot[img['id']]['annotations'] = []

        sent_options = list(range(len(annot['sent_ids'])))
        chosen_idx = random.choice(sent_options)
        new_annot['caption'] = annot['sentences'][chosen_idx]['sent']
        new_annot['bbox'] = coco_style_info[annot['ann_id']]['bbox']
        
        img_id_to_annot[annot['image_id']]['annotations'].append(new_annot)
    
    # remove images which do not belong to any of the train annotations from the full annotations set
    for img_idx in list(img_id_to_annot.keys()):
        if len(img_id_to_annot[img_idx]['annotations']) == 0:
            del img_id_to_annot[img_idx]
    return img_id_to_annot

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

# Function to generate the final jsonl annotations based on the collated full annotations
def generate_jsonl_annotations(full_annotations):
    new_jsonl_entries = []
    
    for img_id, img_info in tqdm(full_annotations.items(), ascii=True, total=len(full_annotations.keys())):
        if 'val2014' in img_info['file_name']:
            image_path = get_coco_path('val', img_info['file_name'], 'images/val2014')
        elif 'train2014' in img_info['file_name']:
            image_path = get_coco_path('train', img_info['file_name'], 'images/train2014')
        else:
            raise Exception("Data must come from one of ['val2014', 'train2014']")

        assert os.path.exists(os.path.join(os.environ['COCO_DIR'], image_path)), 'Image {} does not exist!!!'.format(image_path)
        
        img_width, img_height = img_info['width'], img_info['height']
        
        convs = []
        first_annot_flag = True
        for annot in img_info['annotations']:
            prompt_ = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>".format(annot['caption'])

            reformatted_bbox = normalize_coordinates(convert_bbox_xywh_xyxy(annot['bbox']), img_width, img_height)
            response_ = "<ref>{}</ref><box>[{}]</box>".format(annot['caption'], reformatted_bbox)
            
            # Image is only included in the prompt for the beginning of the conversation, following InternVL 2.5
            if first_annot_flag:
                first_annot_flag = False

                # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                image_beginning = random.choice([0, 1])
                if image_beginning:
                    conv = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":response_}]
                else:
                    conv = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
            else:
                    conv = [{"from":"human", "value":"{}".format(prompt_)}, {"from":"gpt", "value":response_}]

            convs += conv

        entry_ = {"id": len(new_jsonl_entries), "image": image_path, "conversations": convs, "width": img_width, "height": img_height}
        new_jsonl_entries.append(entry_)
    
    return new_jsonl_entries

######## first download the RefCOCO, RefCOCO-G and RefCOCO+ annotations

# create dataset directory if it does not exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# check if the original zip files have already been downloaded/failed to be erased to avoid duplicated downloads
upath_ = output_dir


# RefCOCO
dpath_ = os.path.join(output_dir, 'refcoco.zip')
if not os.path.exists(dpath_) and not os.path.exists(os.path.join(output_dir, 'refcoco')):
    print('Downloading: {}'.format(REFCOCO_ZIP))
    download_file(REFCOCO_ZIP, dpath_)
else:
    print(f"File '{dpath_}' already exists. Skipping download.")

if not os.path.exists(os.path.join(output_dir, 'refcoco')):
    unzip_file(dpath_, upath_)
    print('Cleaned up and ready in: {}!'.format(upath_))
else:
    print(f"Folder '{upath_+'/refcoco'}' already exists. Skipping unzip.")


# RefCOCOG
dpath_ = os.path.join(output_dir, 'refcocog.zip')
if not os.path.exists(dpath_) and not os.path.exists(os.path.join(output_dir, 'refcocog')):
    print('Downloading: {}'.format(REFCOCOG_ZIP))
    download_file(REFCOCOG_ZIP, dpath_)
else:
    print(f"File '{dpath_}' already exists. Skipping download.")

if not os.path.exists(os.path.join(output_dir, 'refcocog')):
    unzip_file(dpath_, upath_)
    print('Cleaned up and ready in: {}!'.format(upath_))
else:
    print(f"Folder '{upath_+'/refcocog'}' already exists. Skipping unzip.")

# RefCOCO+
dpath_ = os.path.join(output_dir, 'refcoco+.zip')
if not os.path.exists(dpath_) and not os.path.exists(os.path.join(output_dir, 'refcoco+')):
    print('Downloading: {}'.format(REFCOCOPLUS_ZIP))
    download_file(REFCOCOPLUS_ZIP, dpath_)
else:
    print(f"File '{dpath_}' already exists. Skipping download.")

if not os.path.exists(os.path.join(output_dir, 'refcoco+')):
    unzip_file(dpath_, upath_)
    print('Cleaned up and ready in: {}!'.format(upath_))
else:
    print(f"Folder '{upath_+'/refcoco+'}' already exists. Skipping unzip.")

######## We now want to start the reformatting procedure


# For RefCOCO, we utilize the 'unc' split, which is the one with the separate testA and testB splits often used in recent models
# This split also follows from Yu et al. https://arxiv.org/pdf/1608.00272
refcoco_coco_style_annots = json.load(open(os.path.join(output_dir, 'refcoco/instances.json'), 'r'))
refcoco_pickle_annots = pickle.load(open(os.path.join(output_dir, 'refcoco/refs(unc).p'), 'rb'))
refcoco_full_annots = get_full_annotations(refcoco_coco_style_annots, refcoco_pickle_annots)
refcoco_jsonl_annots = generate_jsonl_annotations(refcoco_full_annots)

with open(os.path.join(output_dir, 'refcoco_preprocessed.jsonl'), 'w') as f:
    for entry in refcoco_jsonl_annots:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing for refcoco complete! Saved as 'refcoco/refcoco_preprocessed.jsonl' with {} total samples".format(len(refcoco_jsonl_annots)))


# For RefCOCOg, we utilize the 'google' split following Yu et al. https://arxiv.org/pdf/1608.00272
refcocog_coco_style_annots = json.load(open(os.path.join(output_dir, 'refcocog/instances.json'), 'r'))
refcocog_pickle_annots = pickle.load(open(os.path.join(output_dir, 'refcocog/refs(google).p'), 'rb'))
refcocog_full_annots = get_full_annotations(refcocog_coco_style_annots, refcocog_pickle_annots)
refcocog_jsonl_annots = generate_jsonl_annotations(refcocog_full_annots)

with open('./refcoco/refcocog_preprocessed.jsonl', 'w') as f:
    for entry in refcocog_jsonl_annots:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing for refcocog complete! Saved as 'refcoco/refcocog_preprocessed.jsonl' with {} total samples".format(len(refcocog_jsonl_annots)))


# For RefCOCO+, we utilize the 'unc' split, which is the one with the separate testA and testB splits often used in recent models
# This split also follows from Yu et al. https://arxiv.org/pdf/1608.00272
refcocoplus_coco_style_annots = json.load(open(os.path.join(output_dir, 'refcoco+/instances.json'), 'r'))
refcocoplus_pickle_annots = pickle.load(open(os.path.join(output_dir, 'refcoco+/refs(unc).p'), 'rb'))
refcocoplus_full_annots = get_full_annotations(refcocoplus_coco_style_annots, refcocoplus_pickle_annots)
refcocoplus_jsonl_annots = generate_jsonl_annotations(refcocoplus_full_annots)

with open('./refcoco/refcoco+_preprocessed.jsonl', 'w') as f:
    for entry in refcocoplus_jsonl_annots:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing for refcoco+ complete! Saved as 'refcoco/refcoco+_preprocessed.jsonl' with {} total samples".format(len(refcocoplus_jsonl_annots)))

shutil.rmtree('./refcoco/refcoco')
shutil.rmtree('./refcoco/refcocog')
shutil.rmtree('./refcoco/refcoco+')


from tqdm import tqdm
import os
import shutil
import json
import tarfile
import random
random.seed(0)

ROOT_DIR = './objects365'
IMAGE_DIR = './objects365/images'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Download the training annotations
os.system("wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.tar.gz -P ./objects365/")

# Download the training images
for i in range(0, 51):
    os.system("wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/patch"+str(i)+".tar.gz -P ./objects365/images/")

# Untar all of the training patches
for i in range(0, 51):
    with tarfile.open(os.path.join(IMAGE_DIR, 'patch'+str(i)+'.tar.gz')) as tar:
        tar.extractall(path=IMAGE_DIR)

with tarfile.open(os.path.join(ROOT_DIR, 'zhiyuan_objv2_train.tar.gz')) as tar:
    tar.extractall(path=ROOT_DIR)



###### Utility functions

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



###### Read and create preprocessed jsonl file

# Load the existing train json annotations
train_json_annotations = json.load(open(os.path.join(ROOT_DIR, 'zhiyuan_objv2_train.json')))

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

for img_id, img_info in tqdm(img_id_to_annot.items(), ascii=True, total=len(img_id_to_annot.keys())):
    image_path = img_info['file_name'].split('/')[:1] + img_info['file_name'].split('/')[2:]
    image_path = '/'.join(image_path)
    
    # There might be a few deleted images, hence skip them and keep their count
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

print("Preprocessing complete! Saved as 'objects365/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
print("In addition, there were {} missing entries in the downloaded images...".format(len(missing_entries)))


# Remove the tar files after processing...
for i in range(0, 51):
    os.remove(os.path.join(IMAGE_DIR, 'patch'+str(i)+'.tar.gz'))

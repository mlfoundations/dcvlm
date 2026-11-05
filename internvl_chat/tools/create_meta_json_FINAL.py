import os
import random
import json
from pathlib import Path
from tqdm import tqdm
import time
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/datacomp_data")
META_PATH = "./full_run_meta.json"

def compute_length(jsonfile):
    with open(str(jsonfile), 'r', encoding="utf-8") as f:
        for ind, line in enumerate(f): pass
    return ind + 1

with open('all_dataset_names.txt', 'r') as f:
    lines = list(f.readlines())

count = 0
with open('all_dataset_names_and_lengths.txt', 'w') as fout:
    for l in tqdm(lines):
        l = l.strip()
        if l.endswith('.jsonl'):
            length = compute_length(l)
            print(l, length)
            fout.write(f'{l},{length}\n')
        else:
            fout.write(f'{l}\n')
        count += 1
        if count % 100 == 0:
            fout.flush()
            time.sleep(2)

COCO_DIR = os.path.join(DATA_ROOT, "coco")
# datasets whose images live in the shared COCO image folder
COCO_IMAGE_DATASETS = ("coco_caption", "okvqa", "aokvqa", "vsr", "visdial", "cocotext", "refcoco", "as_v2")

def get_dataroot(jsonl):
    if "lvis_instruct_4v" in jsonl:
        return COCO_DIR
    if any(os.path.join(DATA_ROOT, ds) in jsonl for ds in COCO_IMAGE_DATASETS):
        return COCO_DIR
    return os.path.dirname(jsonl)


metadict = {}
with open('all_dataset_names_and_lengths.txt', 'r') as f:
    lines = list(f.readlines())
for l in tqdm(lines):
    path, length = l.strip().split(',')
    root = get_dataroot(path)
    is_wds = False
    if 'detoxify' in root: #for webdataset
        is_wds = True
        root = root.split('detoxify')[0]
        if 'flat' in root:
            root = root.split('flat')[0]
    if not is_wds:
        curr_ = {
            "root": str(root), 
            "annotation": path,
            "data_augment": False, 
            "repeat_time": 1, 
            "length": int(length),
        }
    else:
        curr_ = {
            "root": path,
            "data_augment": False,
            "repeat_time": 1,
            "length": int(length),
            "dataloading_class": "webdataset"
        }
        if 'mint' in path or 'omnicc' in path or 'wanjuan' in path or 'multimodal_textbook' in path:
            curr_["data_type"] = "mmdoc_ic"

    ds_name = str(Path(path).parent.name)
    split_name =  str(Path(path).name)

    if split_name == "preprocessed.jsonl":
        key = ds_name
    else:
        if '/detoxify/sscd' in path:
            if '/flat' in path:
                key = path.split('/flat/')[0].split('/')[-1]
            else:
                key = path.split('/detoxify/')[0].split('/')[-1]
        else:
            key = ds_name + "_" + split_name.removesuffix(".jsonl").replace("preprocessed", '').strip('_')
        key = key.strip('_')
    key = key.replace('_detoxify_sscd', '')    
    metadict[key] = curr_

with open(META_PATH, 'w') as f:
    json.dump(metadict, f, indent=4)

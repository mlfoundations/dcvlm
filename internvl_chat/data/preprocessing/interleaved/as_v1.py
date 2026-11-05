import os
import time
import json
import os.path as osp
from glob import glob
from tqdm import tqdm
from PIL import Image
from functools import partial

import multiprocessing as mp
import webdataset as wds

# yes I know this is bad, but come on
DEBUG_MODE = False
OUT_PATH = os.environ.get("AS_OUT_DIR", "/path/to/as100m/shards/")
# AS-V1 annotations converted to COCO format (expects a `coco_format/` subfolder)
AS_ANNOTATIONS_DIR = os.environ.get("AS_ANNOTATIONS_DIR", "/path/to/AS-100M")
# untarred SA-1B images (see sa1b/untar_multi_sa1b.sh)
SA1B_IMAGES_DIR = os.environ.get("SA1B_IMAGES_DIR", "/path/to/sa1b_raw/")


def load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def dump_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def bname_noext(path: str):
    return osp.splitext(osp.basename(path))[0]


def is_valid(bname: str, root: str):
    expanded_jpg = osp.join(root, bname) + ".jpg"
    expanded_json = osp.join(root, bname) + ".json"
    if osp.exists(expanded_jpg) and osp.exists(expanded_json):
        return True
    return False


def shard_from_dir(path: str):
    assert osp.isdir(path)
    clean = path.removesuffix("/")
    return osp.basename(clean)


def decode_item(k: str, v: str, root: str, ext: str = ".jpg"):
    assert ext.startswith(".")
    fpath = osp.join(root, v, k + ext)
    assert osp.exists(fpath)
    return fpath


def subdir_worker(subdir):
    all_files = glob(osp.join(subdir, '*'))
    unique_bnames = set(bname_noext(f) for f in all_files)
    valid_bnames = [bname for bname in unique_bnames if is_valid(bname, root=subdir)]
    partial_dict = {bname: shard_from_dir(subdir)  for bname in valid_bnames}
    return partial_dict


def normalize_coordinates(box, image_width, image_height):
    # we are using the coco format, so we need to tweak coordinates first
    # NOTE: function taken from InternVL2.5 official data docs
    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x+w, y+h
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box


def map_basenames_to_path(root: str, debug_mode: bool = False, outpath: str = "bname2path.json"): 
    if osp.exists(outpath):
        return load_json(outpath)

    # list shard folders
    subdirs = glob(osp.join(root, "*/"))
    subdirs = [subdir for subdir in subdirs if osp.isdir(subdir)]

    # set a limit for debugging mode 
    limit = len(subdirs) if not debug_mode else 100

    # pool of parallel workers to map images to their shard
    processes, time_in = mp.cpu_count(), time.time()
    print(f"About to open a pool with n={processes} processes.")
    with mp.Pool(processes=processes) as pool:
        partial_dicts = pool.imap_unordered(
            subdir_worker, subdirs[:limit]
        )

        # put everything into a dict, dump and exit
        bname2path = {}
        for partial_dict in tqdm(partial_dicts, total=limit):
            bname2path.update(partial_dict)

    print(f"Mapped a total of {limit} files in {time.time()-time_in:.2f}s")
    dump_json(outpath, bname2path)
    return bname2path


def rearrange_as_dict(annotations: list[dict], key: str = "image_id", key_template: str = "sa_{}"):
    """This functions rearranges annotations of AllSeeing-V1 (coco_format)
    into a dictionary accessed by the image_id within the SA-1B dataset (like "sa_000001").
    Since multiple annotations are possible, each keys is associated to a list of dictionaries, 
    each having the following fields:  
    - bbox: bounding box of the region of interest;  
    - questions: list[str] list of questions; 
    - answers: list[str] of answers replying to questions;"""
    
    outer = {}
    for ann in annotations:
        # this will be the image_id
        key_ = key_template.format(ann[key])
        
        # basically we copy the annotations without the "image_id" key
        ann_ = {k: v for k, v in ann.items() if k != key}

        # make sure each value of the dictionary is a list of annots
        if key_ not in outer:
            outer[key_] = [ann_]
        else:
            outer[key_].append(ann_)
    return outer

def filter_ann(ann: dict, fields: list[str]):
    return {k: ann[k] for k in fields}


def write_tars(
    tar_idx: int,
    bname_mapping: dict, 
    images_dict: dict, 
    annots_dict: dict, 
    images_root: str):

    # this allows reducing to a single arg for all indices so we can 
    # avoid starmap and use imap_unordered which is much faster
    # start, end, tar_idx = indices_tpl

    # expand the tar idx to make it a path
    tarpath = f"{tar_idx:05d}.tar"
    
    # Path to write the resharded tars to
    os.makedirs(OUT_PATH, exist_ok=True)
    sink_pattern = osp.join(OUT_PATH, tarpath)

    with wds.TarWriter(sink_pattern, mtime=1) as writer:
        
        # we are about to write one single item per image, with 
        # multiple annotations in a json file + a global caption
        for bname in tqdm(images_dict.keys(), position=tar_idx, leave=True, desc=f"{tarpath}"):

            # before doing anything, we check if the current basename
            # exists in both annotation dictionaries or else we skip
            if bname not in bname_mapping or bname not in annots_dict: 
                continue

            # if we get here, we have all the ingredient to write a sample
            sample = {}
            sample['__key__'] = bname

            # SA-1B images are quite big, so we will store them in jpg rather
            # than storing the raw byte stream
            subdir = bname_mapping[bname]
            impath = decode_item(bname, subdir, images_root)
            with open(impath, "rb") as stream:
                img = stream.read()
            sample['jpg'] = img

            # global caption generated by MiniGPT-4
            caption = images_dict[bname][0]['caption']
            sample['txt'] = caption

            # save all dense annotations separately, like "0.json", "1.json", etc
            annots = annots_dict[bname]
            image_width = images_dict[bname][0]['width']
            image_height = images_dict[bname][0]['height']
            for i, ann in enumerate(annots):
                # each region annotation has the following keys:
                # - bbox: normalized bounding box coords; 
                # - candidate_list: candidate semantic tags 
                # - candidate_source: 
                # - questions: 3 questions asked by Vicuna given text-only semantic tags; 
                # - answers: respective 3 answers obtained given by Husky; 
                # - region_caption: summarized description of the region given the 3 answers; 
                # --- NOTE: unclear who's the previous summarizer from the paper;
                clean_ann = {
                    'bbox': ann['bbox'], 
                    'questions': ann['attributes']['questions'], 
                    'answers': ann['attributes']['answers'], 
                    'region_caption': ann['region_caption'],
                    'candidate_list': ann['candidate_list'],
                    'candidate_source': ann['candidate_source'],
                    'candidate_clip_logits': ann['candidate_clip_logits']
                }
                clean_ann["bbox"] = normalize_coordinates(
                    clean_ann["bbox"], image_width, image_height
                )
                sample[f'{i}.json'] = clean_ann

            writer.write(sample)
    
    return osp.join(OUT_PATH, tarpath)


def json2tar(i: int, json_files: list, bname_mapping: dict, images_root: str):
    data = load_json(json_files[i])

    images_dict = rearrange_as_dict(
        annotations=data["images"],
        key="image_id"
    )

    annots_dict = rearrange_as_dict(
        annotations=data["annotations"], 
        key="image_id"
    )

    # write a tar corresponding to all images of a json file
    return write_tars(
        tar_idx=i, 
        bname_mapping=bname_mapping,
        images_dict=images_dict,
        annots_dict=annots_dict,
        images_root=images_root,
    )


def main():
    coco_folder = osp.join(AS_ANNOTATIONS_DIR, "coco_format")
    json_files = glob(osp.join(coco_folder, "*.json"))

    time_in = time.time()
    root = SA1B_IMAGES_DIR
    bname2path = map_basenames_to_path(
        root=root, 
        debug_mode=DEBUG_MODE
    )
    print(f"Loaded mapping of {len(bname2path):,} images =) in {time.time()-time_in:.2f}s")


    writing_func = partial(
        json2tar, json_files=json_files, bname_mapping=bname2path, images_root=root
    )

    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.imap_unordered(
            writing_func, list(range(len(json_files)))
        )

        # gotta do sth or bro won't imap
        for r in result:
            continue


if __name__ == "__main__":
    main()

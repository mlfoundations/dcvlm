import os
import json
import time
import argparse
import random
import os.path as osp
import multiprocessing as mp

from glob import glob
from tqdm import tqdm
from functools import partial
from webdataset import TarWriter


def load_jsonl(f: str):
    with open(f, "r", encoding="utf-8") as file_ptr:
        data = [json.loads(s) for s in tqdm(file_ptr)]
    return data


def line2sample(line: dict, images_root: str):
    assert osp.exists(images_root) and osp.isdir(images_root)
    # the InternVL-SA-1B dataset has data formatted in the 'conversations' 
    # format we need already, but some examples are multi-image 
    # NOTE: (however, they are just a single turn conversation and images are not
    # really related to one another, as well as the assistant's response is basically
    # just a concatenation of captions)
    sample = {}
    
    # handle multi-image inputs
    if isinstance(line['image'], list):
        for i, img_path in enumerate(line['image']):
            # write all images as 0.jpg, 1.jpg, etc
            with open(osp.join(images_root, img_path), "rb") as stream:
                img = stream.read()
            sample[f'{i}.jpg'] = img
            
        # write associates texts as 0.txt, 1.txt, etc
        assistant = line['conversations'][1]["value"]
        captions = assistant.split("\n\n")
        
        if len(captions) != len(line['image']):
            raise Exception

        for i, caption in enumerate(captions):
            remove = f"Image-{i+1}: "
            clean = caption.removeprefix(remove)
            sample[f'{i}.txt'] = clean      
    
    # and handle single-image inputs
    else:
        img_path = line['image']
        with open(osp.join(images_root, img_path), "rb") as stream:
            img = stream.read()
        sample['.jpg'] = img
        sample['.txt'] = line['conversations'][1]["value"]

    return sample


def write_shards(input: tuple, images_root: str, dest_folder: str):
    # so we avoid pickling large files
    idx, lines_subset = input

    # define the filepath and some useful variables
    sink_pattern = osp.join(dest_folder, f"{idx:05d}.tar")
    missed = 0 

    # start writing the tar
    with TarWriter(sink_pattern, mtime=1) as writer:  
        for line_idx, line in enumerate(lines_subset):
            # NOTE: some images may be corrupted, if that's the case we simply skip them
            try: 
                sample = line2sample(line, images_root=images_root)
                sample['__key__'] = f"{line_idx:07d}_{line['lan']}"
                writer.write(sample)
            except:
                missed += 1
                continue

    print(f"{sink_pattern} completed. Missed = {missed} samples ({missed/len(lines_subset)*100:.2f}%)")
    return sink_pattern


def main(args):
    random.seed(0)

    # some constants
    images_root = args.images_root
    shards_folder = args.output_dir
    max_samples_per_shard = 10_000
    folder = args.annotations_dir
    
    # load all annots
    files = glob(osp.join(folder, f"*.jsonl"))
    assert len(files) == 4

    # collate all annots (and inject which language they come from)
    lines = []
    for f in files:
        curr_lines = load_jsonl(f)
        for line in curr_lines:
            line['lan'] = 'en' if f.endswith('en.jsonl') else 'zh'
        lines.extend(curr_lines)
    print(f"Loaded a total of {len(lines):,} lines.")

    # make sure they meet the same format
    for l in (lines):
        assert len(l['conversations']) == 2
    print("All checks for formatting have passed :)")

    # shuffle data so lines from same file are not contiguous
    random.shuffle(lines)

    # create inputs for the multprocessing pool
    num_shards = round(len(lines)/max_samples_per_shard + 0.5)
    inputs = []
    for idx in range(num_shards):
        start = max_samples_per_shard*idx
        end = min(len(lines), start+max_samples_per_shard)
        lines_subset = lines[start:end]
        inputs.append((idx, lines_subset)) # tuple!
        print(idx, start, end)

    # make a partial
    writing_func = partial(
        write_shards, 
        images_root=images_root,
        dest_folder=shards_folder,
    )

    # spawn the pool
    time_in = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.imap_unordered(writing_func, inputs)
        for r in result:
            assert osp.exists(r) and osp.isfile(r)

    print(f"Written {num_shards} shards in {time.time()-time_in:.2f} seconds.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_dir", type=str, required=True, help="folder with the InternVL-SA-1B-Caption *.jsonl files")
    parser.add_argument("--images_root", type=str, required=True, help="untarred SA-1B images (see sa1b/untar_multi_sa1b.sh)")
    parser.add_argument("--output_dir", type=str, required=True, help="where to write the shards")
    main(parser.parse_args())
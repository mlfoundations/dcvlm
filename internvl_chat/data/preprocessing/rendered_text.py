from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import io
import random
import re

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import subprocess
import tarfile
import gc
from concurrent.futures import ProcessPoolExecutor

'''
These prompts were generated using GPT-4O using the following prompt:
I am trying to create an OCR dataset for training multimodal models. I currently have an image and a ground-truth parse of all the characters present in the image. For example, this is how the ground-truth parse would look like:
'{"gt_parse": {"text_sequence": "is health had deteriorated so much that he could perf orm only nominal duties. He t ur ne d ov er the pr esid ency to Fuchs after t he season .Death and le gacy.After contractin"}}'
I want to now create a set of prompts from which I can sample for prompting the model to perform the task of OCR. For example, a simple valid prompt would be: "Please read all the characters present in the image." Another valid prompt would be: "Perform the task of optical character recognition and output all the characters that you see in the image, in sequence."
Could you please give a python list of 10-15 prompts which I can use as training prompts for constructing my OCR dataset.
'''
OCR_PROMPTS = [
    "Please read all the characters present in the image.",
    "Perform the task of optical character recognition and output all the characters that you see in the image, in sequence.",
    "Extract and return the complete text sequence from the given image.",
    "Identify and transcribe all the characters present in the image accurately.",
    "Recognize and convert the text present in the image into machine-readable format.",
    "Read and extract the exact sequence of characters from the image without any modifications.",
    "Perform OCR on the image and provide the output as a continuous character sequence.",
    "Retrieve all visible text from the image and present it in a structured format.",
    "Analyze the given image and return the complete sequence of detected characters.",
    "Detect and extract all letters, numbers, and symbols from the image in the correct order.",
    "Carefully read and transcribe every character appearing in the image as it is.",
    "Identify, recognize, and output all text characters in the image while preserving their sequence.",
    "Extract the ground-truth text from the given image using optical character recognition.",
    "Convert the text present in the image into digital form by extracting all characters.",
    "Look at the provided image and return the sequence of text exactly as it appears."
]

def process_tar_file(args):
    tar_path, tar_index, download_dir, output_dir = args
    
    local_jsonl_entries = []
    local_imgs_saved = 0
    local_cnt = 0
    
    extracted_dir = os.path.join(download_dir, os.path.basename(tar_path).replace('.tar', ''))
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir, exist_ok=True)
        
        with tarfile.open(tar_path, "r") as tar_ref:
            tar_ref.extractall(extracted_dir)
    
    json_files = [x for x in os.listdir(extracted_dir) if x.endswith(".json")]
    
    for json_index, json_file in enumerate(json_files):
        try:
            with open(os.path.join(extracted_dir, json_file), 'r') as f:
                data = json.load(f)

            image_path = json_file.replace('.json', '.png')
            full_image_path = os.path.join(extracted_dir, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"Image not found for {json_file}")
                continue

            text = data["ocr_annotation"]["text"]
            # converting to string
            text = ' '.join(text)

            assert os.path.exists(full_image_path) is not None, 'image path is None!!!'
            assert text is not None, 'text is None!!!'

            file_name = f"{output_dir}/images/tar_{tar_index}_sample_{json_index}.png"
            shutil.copyfile(full_image_path, file_name)
            local_imgs_saved += 1

            image_beginning = random.choice([0, 1])
            prompt_ = random.choice(OCR_PROMPTS)
            
            if image_beginning:
                convs = [{"from": "human", "value": "<image>\n{}".format(prompt_)}, 
                         {"from": "gpt", "value": text}]
            else:
                convs = [{"from": "human", "value": "{}\n<image>".format(prompt_)}, 
                         {"from": "gpt", "value": text}]
                
            entry_ = {
                "id": random.randint(10**9, 10**10 - 1),
                "image": 'images/' + os.path.basename(file_name),
                "conversations": convs
            }
            
            local_jsonl_entries.append(entry_)
            local_cnt += 1
            
        except Exception as e:
            print(f"Error processing {json_file} in {tar_path}: {str(e)}")
    

    shutil.rmtree(extracted_dir)
    
    return local_jsonl_entries, local_imgs_saved, local_cnt

def write_jsonl_chunk(entries, output_dir, chunk_number):
    """Write a chunk of entries to a JSONL file"""
    output_file = os.path.join(output_dir, f"preprocessed_part_{chunk_number}.jsonl")
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    return output_file

def main():
    download_dir = "./renderedtext_tars"
    output_dir = "./renderedtext"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    snapshot_download(
        repo_id="wendlerc/RenderedText",
        repo_type="dataset",
        local_dir=download_dir)

    new_jsonl_entries = []
    cnt = 0
    imgs_saved = 0
    
    tars_main = [os.path.join(download_dir, x) for x in os.listdir(download_dir) if x.endswith(".tar")]
    tars_remaining = [os.path.join(download_dir, "remaining", x) for x in
                     os.listdir(os.path.join(download_dir, "remaining")) if x.endswith(".tar")]
    tars = tars_main + tars_remaining
        
    # For parallel processing, we consider each task processes one tar file completely as the tars are not too large
    tasks = [(tar, tar_index, download_dir, output_dir) for tar_index, tar in enumerate(tars)]
    
    print(f"Processing {len(tasks)} TAR files in parallel...")
    
    # reduce max_workers to avoid overloading the system
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(
            executor.map(process_tar_file, tasks),
            total=len(tasks),
            desc="Processing TAR files"
        ))
    
    new_jsonl_entries = []
    imgs_saved = 0
    
    for local_entries, local_imgs_saved, local_cnt in results:
        new_jsonl_entries.extend(local_entries)
        imgs_saved += local_imgs_saved
    
    print("Sorting entries...")
    new_jsonl_entries.sort(key=lambda x: x["id"])
    
    entries_per_file = 1000000
    total_entries = len(new_jsonl_entries)
    num_files = (total_entries + entries_per_file - 1) // entries_per_file
    
    print(f"Total images saved: {imgs_saved}")
    print(f"Total entries: {total_entries}")
    print(f"Splitting into {num_files} files with {entries_per_file} entries each...")
    
    output_files = []
    
    for i in range(num_files):
        start_idx = i * entries_per_file
        end_idx = min((i + 1) * entries_per_file, total_entries)
        
        chunk = new_jsonl_entries[start_idx:end_idx]
        for j, entry in enumerate(chunk):
            entry["id"] = j
        
        output_file = write_jsonl_chunk(chunk, output_dir, i + 1)
        output_files.append(output_file)
        
        print(f"Written {len(chunk)} entries to {output_file}")

    metadata = {
        "total_entries": total_entries,
        "total_images": imgs_saved,
        "entries_per_file": entries_per_file,
        "number_of_files": num_files,
        "files": output_files
    }
    
    with open(os.path.join(output_dir, "preprocessing_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    gc.collect()

if __name__ == "__main__":
    main()

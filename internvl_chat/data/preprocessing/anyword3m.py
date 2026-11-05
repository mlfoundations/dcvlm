import os
import json
import random
import pandas as pd
import io
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import time
import numpy as np

# random.seed(0)

assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

LLAVA_DESC_SHORT_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo’s key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]

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
    "Look at the provided image and return the sequence of text exactly as it appears."]

BUFFER_SIZE = 1000  

def save_image(image, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image.save(file_path)
        return True
    except Exception as e:
        print(f"Error saving image to {file_path}: {e}")
        return False
    finally:
        if image is not None:
            image.close()

def process_row(row_data):
    row, folder, parquet_index, sample_index, output_dir = row_data
    
    try:
        if folder == "OCR_COCO_Text":
            image_path_old = row["img_name"]
            file_name = os.path.join(os.environ['COCO_DIR'], "images/train2014/train2014", image_path_old)
            height = row["image_height"]
            width = row["image_width"]
            response_ = row["caption"]
            prompt_ = random.choice(LLAVA_DESC_SHORT_PROMPTS)
        
        elif folder in ["OCR_Art", "OCR_ReCTS", "OCR_MTWI2018", "OCR_icdar2017rctw", 
                      "OCR_mlt2019", "laion", "wukong_1of5", "wukong_2of5", 
                      "wukong_3of5", "wukong_4of5", "wukong_5of5"]:
            image_bytes = row["image"]["bytes"]
            image = Image.open(io.BytesIO(image_bytes))
            file_name = f"{output_dir}/images/{folder}_parquet_{parquet_index}_sample_{sample_index}.png"
            save_image(image, file_name)
            image.close()
            image = None
            
            height = row["image_height"]
            width = row["image_width"]
            caption = row["caption"]
            
            if "annotations" not in row or row["annotations"] is None:
                response_ = caption
                prompt_ = random.choice(LLAVA_DESC_SHORT_PROMPTS)
            else:
                ocr_text = ", ".join(entry["text"] for entry in row["annotations"])
                response_ = "\n".join(["Caption: " + caption, "Text: " + ocr_text])
                prompt_ = f"First, {random.choice(LLAVA_DESC_SHORT_PROMPTS)} Then, {random.choice(OCR_PROMPTS)}."
        
        elif folder == "OCR_LSVT":
            image_bytes = row["image"]["bytes"]
            image = Image.open(io.BytesIO(image_bytes))
            file_name = f"{output_dir}/images/{folder}_parquet_{parquet_index}_sample_{sample_index}.png"
            save_image(image, file_name)
            image = None
            
            height = row["image_height"]
            width = row["image_width"]
            response_ = row["caption"]
            prompt_ = random.choice(LLAVA_DESC_SHORT_PROMPTS)
        else:
            return None

        image_beginning = random.choice([0, 1])
        if image_beginning:
            convs = [{"from": "human", "value": "<image>\n{}".format(prompt_)}, 
                     {"from": "gpt", "value": response_}]
        else:
            convs = [{"from": "human", "value": "{}\n<image>".format(prompt_)}, 
                     {"from": "gpt", "value": response_}]
            
        entry_ = {
            "id": random.randint(10**9, 10**10 - 1),
            "image": 'images/' + os.path.basename(file_name),
            "height": height,
            "width": width,
            "conversations": convs
        }

        if folder == "laion":
            return ("laion", entry_)
        elif "wukong" in folder:
            return ("wukong", entry_)
        else:
            return ("rest", entry_)
            
    except Exception as e:
        print(f"Error processing {folder}, parquet {parquet_index}, sample {sample_index}: {e}")
        return None

def write_entries_to_file(entries, output_dir):
    rest_entries = []
    laion_entries = []
    wukong_entries = []
    
    for entry_type, entry in entries:
        if entry_type == "rest":
            rest_entries.append(entry)
        elif entry_type == "laion":
            laion_entries.append(entry)
        elif entry_type == "wukong":
            wukong_entries.append(entry)
    
    if rest_entries:
        with open(f'{output_dir}/ocr_preprocessed.jsonl', 'a') as f:
            for entry in rest_entries:
                f.write(json.dumps(entry) + '\n')
    
    if laion_entries:
        with open(f'{output_dir}/laion_preprocessed.jsonl', 'a') as f:
            for entry in laion_entries:
                f.write(json.dumps(entry) + '\n')
    
    if wukong_entries:
        with open(f'{output_dir}/wukong_preprocessed.jsonl', 'a') as f:
            for entry in wukong_entries:
                f.write(json.dumps(entry) + '\n')

def process_laion_row_group(row_group_data):
    pf_path, row_group_idx, parquet_index, output_dir, batch_size = row_group_data
    entry_buffer = []
    
    try:
        pf = pq.ParquetFile(pf_path)
        table = pf.read_row_group(row_group_idx)
        df = table.to_pandas()
        table = None
        
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            row_data_batch = []
            for sample_index, row in batch_df.iterrows():
                absolute_index = row_group_idx * 10000 + batch_start + sample_index
                row_data = (row, "laion", parquet_index, absolute_index, output_dir)
                row_data_batch.append(row_data)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_row, row_data_batch))
            
            valid_results = [r for r in results if r is not None]
            
            if valid_results:
                write_entries_to_file(valid_results, output_dir)
            
            batch_df = None
            row_data_batch = None
            results = None
            valid_results = None
            gc.collect()
        
        df = None
        gc.collect()
        return True
        
    except Exception as e:
        print(f"Error processing row group {row_group_idx} in laion parquet {parquet_index}: {e}")
        if entry_buffer:
            write_entries_to_file(entry_buffer, output_dir)
        return False
        
def process_parquet_file(args):
    folder, parquet_file, parquet_index, output_dir = args
    parquet_path = os.path.join(output_dir, '..', 'anyword3m_files', folder, parquet_file)
    
    entry_buffer = []
    start_time = time.time()
    
    try:
        if folder == "laion":
            print(f"Processing large file: {parquet_file}")
            
            pf = pq.ParquetFile(parquet_path)
            num_row_groups = pf.num_row_groups
            batch_size = 50
            
            row_group_tasks = []
            for row_group_idx in range(num_row_groups):
                row_group_tasks.append((parquet_path, row_group_idx, parquet_index, output_dir, batch_size))
            
            max_workers = min(4, num_row_groups)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(process_laion_row_group, row_group_tasks), total=len(row_group_tasks)))
            
            gc.collect()
        else:
            df = pd.read_parquet(parquet_path)
            batch_size = 500
            
            for batch_start in tqdm(range(0, len(df), batch_size), ascii=True):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]
                
                row_data_batch = []
                for sample_index, row in batch_df.iterrows():
                    row_data = (row, folder, parquet_index, sample_index, output_dir)
                    row_data_batch.append(row_data)
                
                with ThreadPoolExecutor(max_workers=16) as executor:
                    results = list(executor.map(process_row, row_data_batch))
                
                valid_results = [r for r in results if r is not None]
                entry_buffer.extend(valid_results)
                
                if len(entry_buffer) >= BUFFER_SIZE:
                    write_entries_to_file(entry_buffer, output_dir)
                    entry_buffer = []
                
                batch_df = None
                gc.collect()
            
            df = None
            gc.collect()
            
        if entry_buffer:
            write_entries_to_file(entry_buffer, output_dir)
        
        elapsed_time = time.time() - start_time
        print(f"Processed {parquet_file} in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error processing file {parquet_path}: {e}")
        
        if entry_buffer:
            write_entries_to_file(entry_buffer, output_dir)
        
        return False

def main():
    snapshot_download(
        repo_id="stzhao/AnyWord-3M",
        repo_type="dataset",
        local_dir=download_dir,
    )
    download_dir = "./anyword3m_files"
    output_dir = "./anyword3m"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    for file_name in ['ocr_preprocessed.jsonl', 'laion_preprocessed.jsonl', 'wukong_preprocessed.jsonl']:
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass
    
    folders = [x for x in os.listdir(download_dir) 
              if os.path.isdir(os.path.join(download_dir, x)) and x != ".cache"]
    
    if "laion" in folders:
        folders.remove("laion")
        laion_folder = "laion"
        
        print(f"Processing folder: {laion_folder}")
        parquets = [x for x in os.listdir(f"{download_dir}/{laion_folder}") if x.endswith(".parquet")]
        
        for idx, parquet in enumerate(parquets):
            args = (laion_folder, parquet, idx, output_dir)
            process_parquet_file(args)
            gc.collect()

    for folder in tqdm(folders, desc="Processing folders"):
        print(f"Processing folder: {folder}")
        
            
        parquets = [x for x in os.listdir(f"{download_dir}/{folder}") if x.endswith(".parquet")]
            
        tasks = [(folder, parquet, idx, output_dir) for idx, parquet in enumerate(parquets)]
            
        with ProcessPoolExecutor(max_workers=16) as executor:
            list(tqdm(executor.map(process_parquet_file, tasks), total=len(tasks), 
                    desc=f"Processing {folder} parquets"))
            
        gc.collect()
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()

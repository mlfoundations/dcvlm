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

output_dir = "./synthdog"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+"/images", exist_ok=True)

# naver-clova-ix/synthdog-en
snapshot_download(
    repo_id="naver-clova-ix/synthdog-en",
    repo_type="dataset",
    allow_patterns=["data/train*parquet"],
    local_dir=os.path.join(output_dir, "synthdog_en_parquets")
)

# naver-clova-ix/synthdog-zh
snapshot_download(
    repo_id="naver-clova-ix/synthdog-zh",
    repo_type="dataset",
    allow_patterns=["data/train*parquet"],
    local_dir=os.path.join(output_dir, "synthdog_zh_parquets")
)

# naver-clova-ix/synthdog-ja
snapshot_download(
    repo_id="naver-clova-ix/synthdog-ja",
    repo_type="dataset",
    allow_patterns=["data/train*parquet"],
    local_dir=os.path.join(output_dir, "synthdog_ja_parquets")
)

# naver-clova-ix/synthdog-ko
snapshot_download(
    repo_id="naver-clova-ix/synthdog-ko",
    repo_type="dataset",
    allow_patterns=["data/train*parquet"],
    local_dir=os.path.join(output_dir, "synthdog_ko_parquets")
)


####### First, convert Synthdog-en
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "synthdog_en_parquets", "data")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "synthdog_en_parquets", "data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image = row["image"]
        ocr = json.loads(row["ground_truth"])["gt_parse"]["text_sequence"]
        assert 'bytes' in image, "image does not have bytes!!!"
        assert ocr is not None, "ocr is None!!!"
        # save image
        img_bytes = image["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/synthdog_en_parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1
        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        # add entry to new jsonl list
        prompt_ = random.choice(OCR_PROMPTS)
        response_ = ocr
        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join("images", os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "en_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "synthdog_en_parquets"))

print("Preprocessing complete! Saved as 'synthdog/en_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))




####### Next, convert Synthdog-zh
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "synthdog_zh_parquets", "data")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "synthdog_zh_parquets", "data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image = row["image"]
        ocr = json.loads(row["ground_truth"])["gt_parse"]["text_sequence"]
        assert 'bytes' in image, "image does not have bytes!!!"
        assert ocr is not None, "ocr is None!!!"
        # save image
        img_bytes = image["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/synthdog_zh_parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1
        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        # add entry to new jsonl list
        prompt_ = random.choice(OCR_PROMPTS)
        response_ = ocr
        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join("images", os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "zh_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "synthdog_zh_parquets"))

print("Preprocessing complete! Saved as 'synthdog/zh_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))




####### Next, convert Synthdog-ja
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "synthdog_ja_parquets", "data")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "synthdog_ja_parquets", "data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image = row["image"]
        ocr = json.loads(row["ground_truth"])["gt_parse"]["text_sequence"]
        assert 'bytes' in image, "image does not have bytes!!!"
        assert ocr is not None, "ocr is None!!!"
        # save image
        img_bytes = image["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/synthdog_ja_parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1
        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        # add entry to new jsonl list
        prompt_ = random.choice(OCR_PROMPTS)
        response_ = ocr
        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join("images", os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "ja_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "synthdog_ja_parquets"))

print("Preprocessing complete! Saved as 'synthdog/ja_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))



####### Finally, convert Synthdog-ko
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "synthdog_ko_parquets", "data")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "synthdog_ko_parquets", "data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image = row["image"]
        ocr = json.loads(row["ground_truth"])["gt_parse"]["text_sequence"]
        assert 'bytes' in image, "image does not have bytes!!!"
        assert ocr is not None, "ocr is None!!!"
        # save image
        img_bytes = image["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/synthdog_ko_parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1
        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        # add entry to new jsonl list
        prompt_ = random.choice(OCR_PROMPTS)
        response_ = ocr
        image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
        if image_beginning:
            convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
        else:
            convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join("images", os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, "ko_preprocessed.jsonl"), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "synthdog_ko_parquets"))

print("Preprocessing complete! Saved as 'synthdog/ko_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))

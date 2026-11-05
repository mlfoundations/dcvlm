import os
import json
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
from huggingface_hub import snapshot_download
from PIL import Image
import shutil
import pandas as pd
from tqdm import tqdm
import io
import argparse

# Prompt templates are taken from llava-onevision 9K WebSight samples
TEMPLATES = [
    "Outline the HTML required to reproduce this website's appearance.",
    'Generate the HTML code corresponding to this website screenshot.',
    'Transform this website screenshot into HTML code.',
    'Produce the HTML markup to recreate the visual appearance of this website.',
    "Write the HTML that mirrors this website's layout.",
    "Derive the HTML code to reflect this website's interface.",
    'Compose the HTML code to achieve the same design as this screenshot.',
    "Craft the HTML code that would generate this website's look.",
    "Synthesize the HTML to emulate this website's layout.",
    "Formulate the HTML to replicate this web page's design.",
    'Render the HTML code that corresponds to this web design.',
    "Develop the HTML structure to match this website's aesthetics.",
    "Encode this website's visual representation into HTML.",
    "Illustrate the HTML coding for this website's visual format.",
    'Reconstruct the HTML code from this website image.',
    'Translate this website image into its HTML code.',
    'Convert this screenshot into its equivalent HTML structure.',
    "Assemble the HTML code to mimic this webpage's style."
]

def main(start_parquet_index, end_parquet_index):

    output_dir = "./websight"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/images', exist_ok=True)

    snapshot_download(
        repo_id="HuggingFaceM4/WebSight",
        repo_type="dataset",
        allow_patterns=["v0.2/train*.parquet"],
        local_dir=output_dir
    )

    output_data = []
    imgs_saved = 0
    fils = os.listdir(f'{output_dir}/v0.2')

    for parquet_index, filename in tqdm(enumerate(fils), ascii=True, total=len(fils)):

        # skip parquets based on offsets
        if parquet_index < start_parquet_index:
            continue
        if parquet_index > end_parquet_index:
            continue

        df = pd.read_parquet(f'{output_dir}/v0.2/{filename}')

        for sample_index, row in tqdm(df.iterrows(), ascii=True):
            img_bytes = row['image']['bytes']
            answer = row['text']
            image = Image.open(io.BytesIO(img_bytes))
            file_name = os.path.join('images', f'image_{start_parquet_index}_{end_parquet_index}_{imgs_saved}.png')
            width, height = image.size
            image.save(os.path.join(output_dir, file_name))
            imgs_saved += 1
            image_beginning = random.choice([0, 1])
            prompt = random.choice(TEMPLATES)
            if image_beginning:
                convs = [{"from": "human", "value": "<image>{}".format(prompt)}, {"from": "gpt", "value": answer}]
            else:
                convs = [{"from": "human", "value": "{}<image>".format(prompt)}, {"from": "gpt", "value": answer}]
            
            # make sure ids are unique by writing parquet specific ids
            entry = {"id": imgs_saved + parquet_index * 2700, "image": file_name, "height": height, "width": width, "conversations": convs}
            output_data.append(entry)

    with open(f'{output_dir}/preprocessed_start{start_parquet_index}_end{end_parquet_index}.jsonl', 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

    shutil.rmtree('./websight/v0.2')
    print(f"Preprocessing complete! Saved as 'websight/preprocessed_start{start_parquet_index}_end{end_parquet_index}.jsonl' with {imgs_saved} saved images and {len(output_data)} total samples.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_parquet_index', type=int, default=0, help='Parquet to start processing from')
    parser.add_argument('--end_parquet_index', type=int, default=740, help='Parquet to end processing at')
    args = parser.parse_args()

    main(args.start_parquet_index, args.end_parquet_index)
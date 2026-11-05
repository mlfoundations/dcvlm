import json
import os
import io
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import shutil
import zipfile
import requests

URL_ = "https://math-qa.github.io/math-QA/data/MathQA.zip"

###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

output_dir = "./mathqa"
os.makedirs(output_dir, exist_ok=True)

download_file(URL_, os.path.join(output_dir, "mathqa.zip"))
print("Unzipping!")
zipfile_location = os.path.join(output_dir, "mathqa.zip")
with zipfile.ZipFile(zipfile_location, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(output_dir, "mathqa_uncompressed"))
    print(f"Extracted zip!")

# Delete the zip file
os.remove(zipfile_location)

# Read and create preprocessed jsonl
new_jsonl_entries = []
with open(os.path.join(output_dir, "mathqa_uncompressed", "train.json")) as f:
    contents = json.load(f)

for sample_index, row in tqdm(enumerate(contents), ascii=True, total=len(contents)):

    prompt_ = ''
    prompt_ += 'Question: {}\n'.format(row["Problem"])
    prompt_ += 'Options:\n'
    prompt_ += row["options"]

    answer_ = ''
    if row["Rationale"] is None or len(row["Rationale"]) == 0:
        answer_ = 'Final Answer: {}'.format(row["correct"])
    else:    
        answer_ = '{}\nFinal Answer: {}'.format(row["Rationale"], row["correct"])

    convs = [{"from":"human", "value":"{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    entry_ = {"id":len(new_jsonl_entries), "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./mathqa/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'mathqa/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

shutil.rmtree('./mathqa/mathqa_uncompressed')
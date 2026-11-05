"""
Download training data of the iNaturalist 2018 competition.
https://github.com/visipedia/inat_comp/blob/master/2018/README.md
"""

import os
import random
import subprocess
import shutil
import requests
import tarfile
import json
from tqdm import tqdm

random.seed(0)

OUTPUT_DIR = "./inaturalist2018"

DESCRIPTION_PROMPTS = [
    { "question": "What is the kingdom and phylum of this organism?", "answer": "The kingdom is {kingdom} and the phylum is {phylum}." },
    { "question": "To which class and order does this species belong?", "answer": "The class is {class} and the order is {order}." },
    { "question": "Can you identify the family and genus of this organism?", "answer": "The family is {family} and the genus is {genus}." },
    { "question": "What is the full taxonomic classification of this species from kingdom to genus?", "answer": "Kingdom: {kingdom}, Phylum: {phylum}, Class: {class}, Order: {order}, Family: {family}, Genus: {genus}." },
    { "question": "Which supercategory and family does this species belong to?", "answer": "The supercategory is {supercategory} and the family is {family}." },
    { "question": "What order and genus are associated with this organism?", "answer": "The order is {order} and the genus is {genus}." },
    { "question": "In which phylum and class can this species be classified?", "answer": "The phylum is {phylum} and the class is {class}." },
    { "question": "To which kingdom and supercategory does this species belong?", "answer": "The kingdom is {kingdom} and the supercategory is {supercategory}." },
    { "question": "What family and order does this organism belong to?", "answer": "The family is {family} and the order is {order}." },
    { "question": "What is the genus of this species, and which class does it belong to?", "answer": "The genus is {genus} and the class is {class}." },
    { "question": "Which phylum and order does this organism belong to?", "answer": "The phylum is {phylum} and the order is {order}." },
    { "question": "What are the kingdom and genus of this species?", "answer": "The kingdom is {kingdom} and the genus is {genus}." },
    { "question": "Can you provide the supercategory and class of this organism?", "answer": "The supercategory is {supercategory} and the class is {class}." },
    { "question": "What is the relationship between the family and genus of this species?", "answer": "The family is {family}, and the genus within this family is {genus}." },
    { "question": "Which class does this species belong to, and what is its supercategory?", "answer": "The class is {class} and the supercategory is {supercategory}." },
    { "question": "What kingdom, phylum, and class classify this organism?", "answer": "Kingdom: {kingdom}, Phylum: {phylum}, Class: {class}." },
    { "question": "To which genus and family does this organism belong?", "answer": "The genus is {genus} and the family is {family}." },
    { "question": "What supercategory and phylum is this species part of?", "answer": "The supercategory is {supercategory} and the phylum is {phylum}." },
    { "question": "Can you identify the order and supercategory of this organism?", "answer": "The order is {order} and the supercategory is {supercategory}." },
    { "question": "What is the scientific classification from family to genus for this species?", "answer": "The family is {family} and the genus is {genus}." }
]
"""
Description prompts are generated using GPT-4o with this prompt:
> I have a dataset with images and category information of each image:
> ```
> category{
>   "id" : int,
>   "name" : str,
>   "supercategory" : str,
>   "kingdom" : str,
>   "phylum" : str,
>   "class" : str,
>   "order" : str,
>   "family" : str,
>   "genus" : str
> }
> ```
> Can you create a set of questions I can ask given these categories?
> The output format should be
> ```
> [
>    { "question": "What genus and family does this animal have?", "answer": "The family is {family} and the genus is {genus}" },
> ...
> ]
> ```
> This was just an example, make sure the questions make biological sense.
"""

train_images = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"
train_annotations = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz"
unobfuscated_categories = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/categories.json.tar.gz"


def download_and_extract(url, output_dir, expected_md5: str | None = None):
    local_filename = os.path.join(output_dir, url.split("/")[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    if expected_md5:
        print("Extracting MD5!")
        md5cmd = subprocess.run(
            f"md5sum {local_filename}",
            shell=True,
            capture_output=True,
            check=True,
            text=True,
        )
        assert md5cmd.stdout.split()[0] == expected_md5, "md5sum does not match"

    print("Extracting zip!")
    with tarfile.open(local_filename, "r:gz") as f:
        f.extractall(output_dir)
    os.remove(local_filename)

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
download_and_extract(train_images, OUTPUT_DIR, "b1c6952ce38f31868cc50ea72d066cc3")
print("Moving images!")
shutil.move(f"{OUTPUT_DIR}/train_val2018", f"{OUTPUT_DIR}/images")
download_and_extract(train_annotations, OUTPUT_DIR, "bfa29d89d629cbf04d826a720c0a68b0")
download_and_extract(unobfuscated_categories, OUTPUT_DIR, "191b9b78bff639764b0141deb59e60c8")

with open(os.path.join(OUTPUT_DIR, "train2018.json"), "r") as f:
    data = json.load(f)

with open(os.path.join(OUTPUT_DIR, "categories.json"), "r") as f:
    categories = json.load(f)

categories = {
    c["id"]: c for c in categories
}
annotations = {
    a["image_id"]: a for a in data["annotations"]
}
def get_category_by_image_id(image_id: int):
    cid = annotations[image_id]["category_id"]
    return categories[cid]

image_categories = []
for image in tqdm(data["images"], desc="Processing images"):
    category = get_category_by_image_id(image["id"])
    image_path = image["file_name"].replace("train_val2018/", "images/train_val2018/")
    assert os.path.exists(os.path.join(OUTPUT_DIR, image_path)), "Image in dataset does not exist on filesystem"
    image_categories.append({
        "image_id": image["id"],
        "file_name": image_path,
        "category": category,
        "height": image["height"],
        "width": image["width"],
    })

preprocessed = []
for item in tqdm(image_categories, desc="Creating preprocessed dataset"):
    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    value_str = "<image>{}" if image_beginning else "{}<image>"
    t = random.choice(DESCRIPTION_PROMPTS)
    # randomly pick a prompt
    convs = [
        {"from": "human", "value": value_str.format(t["question"])},
        {"from": "gpt", "value": t["answer"].format(**item["category"])},
    ]
    entry_ = {
        "id": item['image_id'],
        "image": item["file_name"],
        "height": item["height"],
        "width": item["width"],
        "conversations": convs,
    }
    preprocessed.append(entry_)

with open(os.path.join(OUTPUT_DIR, "preprocessed.jsonl"), "w") as f:
    for entry in preprocessed:
        f.write(json.dumps(entry) + "\n")

os.remove(os.path.join(OUTPUT_DIR, "train2018.json"))
os.remove(os.path.join(OUTPUT_DIR, "categories.json"))

print(
    f"Downloaded iNaturalist 2018 competition dataset to {OUTPUT_DIR}"
)

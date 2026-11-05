import json
import os
import io
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import shutil
import zipfile
import requests

TEXTCAPS_URL = 'https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json'
TEXTCAPS_IMAGES_URL = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'

# taken from: https://arxiv.org/pdf/2304.08485 tab. 11
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

output_dir = './textcaps'
os.makedirs(output_dir, exist_ok=True)

###### Utility functions

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")

def format_image_classes(image_classes):

    assert image_classes is not None, 'image_classes is None!!'

    # Convert all elements to lowercase
    image_classes = [cls.lower() for cls in image_classes]
    
    # Construct the sentence based on the number of elements
    if len(image_classes) == 1:
        formatted_sentence = f"This image mainly consists of {image_classes[0]}."
    elif len(image_classes) == 2:
        formatted_sentence = f"This image mainly consists of {image_classes[0]} and {image_classes[1]}."
    else:
        formatted_sentence = f"This image mainly consists of {', '.join(image_classes[:-1])} and {image_classes[-1]}."
    
    return formatted_sentence

def reformat(caption):
    if '.' != caption[-1]:
        return caption+'.'
    return caption

###### Download the files and unzip

# Download caps
train_path = os.path.join(output_dir, "train.json")
download_file(TEXTCAPS_URL, train_path)

# Download and extract images
train_images_path = os.path.join(output_dir, "images.zip")
download_file(TEXTCAPS_IMAGES_URL, train_images_path)
unzip_file(train_images_path, output_dir)

os.rename(os.path.join(output_dir, 'train_images'), os.path.join(output_dir, 'images'))

###### Now preprocess into correct jsonl format

with open(os.path.join(output_dir, 'train.json')) as f:
    train_contents = json.load(f)['data']

new_jsonl_contents = []

for index, contents in tqdm(enumerate(train_contents), ascii=True, total=len(train_contents)):
    image_path = contents['image_path']
    width = contents['image_width']
    height = contents['image_height']
    caption = contents['caption_str']
    image_classes = contents['image_classes']

    assert image_path is not None, 'image_path is None!!!'
    assert caption is not None, 'caption is None!!!'
    assert height is not None, 'height is None!!'
    assert width is not None, 'width is None!!'

    assert os.path.exists(os.path.join(output_dir, 'images', os.path.basename(image_path))), 'image_path does not exist!!'

    prompt_ = random.choice(LLAVA_DESC_SHORT_PROMPTS)

    image_classes_in_beginning = random.choice([0, 1])
    if image_classes_in_beginning:
        response_ = '{} {}'.format(format_image_classes(image_classes), reformat(caption))
    else:
        response_ = '{} {}'.format(reformat(caption), format_image_classes(image_classes))

    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])

    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]

    entry_ = {"id":len(new_jsonl_contents), "image": 'images/'+os.path.basename(image_path), "height": height, "width": width, "conversations": convs}
    new_jsonl_contents.append(entry_)

with open('./textcaps/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_contents:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'textcaps/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_contents)))

os.remove('./textcaps/train.json')
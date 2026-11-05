import os
import kagglehub
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import zipfile
import scipy.io

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

assert 'KAGGLE_USERNAME' in os.environ, 'Please set-up your KAGGLE_USERNAME env variable for your kaggle username, set it using export KAGGLE_USERNAME=XXXX'
assert 'KAGGLE_KEY' in os.environ, 'Please set-up your KAGGLE_KEY env variable for your kaggle key, set it using export KAGGLE_KEY=XXXX'

output_dir = "./synthtext"
os.makedirs(output_dir, exist_ok=True)
# this is to ensure we download to the correct output directory
os.environ["KAGGLEHUB_CACHE"] = output_dir

# Download latest version
path = kagglehub.dataset_download("wassefy/synthtext")

# Load annotations
annotations = scipy.io.loadmat(os.path.join(output_dir, "datasets/wassefy/synthtext/versions/1/SynthText", "gt.mat"))
image_names = annotations["imnames"][0]
text_annots = annotations["txt"][0]

new_jsonl_entries = []

for img_name, text_annot in tqdm(zip(image_names, text_annots), ascii=True, total=len(image_names)):

    img_name = img_name[0]

    # verify image exists
    assert os.path.exists(os.path.join(output_dir, "datasets/wassefy/synthtext/versions/1/SynthText", img_name)), f"{img_name} does not exist!!!"

    # add entry to new jsonl list
    prompt_ = random.choice(OCR_PROMPTS)
    response_ = "".join(list(text_annot))
    final_img_path = os.path.join("datasets/wassefy/synthtext/versions/1/SynthText", img_name)
    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    entry_ = {"id":len(new_jsonl_entries), "image": final_img_path, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

os.remove(os.path.join(output_dir, "datasets/wassefy/synthtext/versions/1/SynthText", "gt.mat"))

print("Preprocessing complete! Saved as 'synthtext/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

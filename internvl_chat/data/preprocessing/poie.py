import gdown
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # ensure reproducibility
import json
import shutil
import zipfile

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

output_dir = "./poie"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+"/images", exist_ok=True)

########## Utility functions

# Function to unzip files
def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")


# Google Drive file ID extracted from the link: https://drive.google.com/file/d/1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs/view
file_id = "1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs"
output_filename = os.path.join(output_dir, "poie.zip")

# Download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_filename, quiet=False)

# Unzipping the file
unzip_file(output_filename, output_dir)

# Move files around
source_dir = os.path.join(output_dir, "nfv5/nfv5_3125/image_files")
destination_dir = os.path.join(output_dir, "images")

# Ensure the source directory exists
if os.path.exists(source_dir) and os.path.isdir(source_dir):
    # Move all files from source to destination
    fils = os.listdir(source_dir)
    for file_name in tqdm(fils, ascii=True, total=len(fils)):
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)

        if os.path.isfile(source_path):  # Only move files, not subdirectories
            shutil.move(source_path, destination_path)
    print("All files moved successfully.")
else:
    print("Source directory does not exist.")

# Create preprocessed jsonl file
new_jsonl_entries = []

with open(os.path.join(output_dir, "nfv5", "nfv5_3125", "train.txt")) as f:
    contents = f.readlines()


for index, content in tqdm(enumerate(contents), ascii=True, total=len(contents)):
    content = json.loads(content)
    image_path = os.path.basename(content["file_name"])
    height = content["height"]
    width = content["width"]
    annotations = content["annotations"]

    assert os.path.exists(os.path.join(output_dir, "images", image_path)), "{} image does not exist!!!".format(image_path)

    prompt_ = random.choice(OCR_PROMPTS)
    response_ = " ".join([x["text"] for x in annotations])

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    entry_ = {"id":len(new_jsonl_entries), "image": 'images/'+image_path, "height": height, "width": width, "conversations": convs}
    new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "nfv5"))

print("Preprocessing complete! Saved as 'poie/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

import json
import os
import io
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import shutil
import tarfile
import requests
import dataclasses
import pprint
import re
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.patches as mpl_patches
from xml.etree import ElementTree
from multiprocessing import Pool, cpu_count

DOWNLOAD_URL = 'https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz'
NUM_WORKERS = cpu_count()

'''
Prompts generated with ChatGPT-4o with the following prompt:

I am trying to construct a synthetic dataset for training a large multimodal model where the dataset contains images of math expressions 
written in latex and the model has to output the latex code required to generate such an image. Could you give me 10-15 prompts that I 
can use as random prompts for the model to construct my synthetic dataset that would bring some diversity in the mix. An example of an 
appropriate prompt would be: "Give the latex snippet required to generate this handwritten math expression".
Give it to me as a python list.
'''
PROMPTS = [
    "Give the LaTeX snippet required to generate this handwritten math expression.",
    "What is the LaTeX code for the mathematical formula shown in the image?",
    "Convert this handwritten equation to its corresponding LaTeX code.",
    "Write the LaTeX expression that represents this math image.",
    "Identify and provide the LaTeX code for the expression in the image.",
    "Translate this handwritten mathematical notation into LaTeX.",
    "Generate the LaTeX code for the formula displayed.",
    "What LaTeX command would reproduce this math expression?",
    "Write the LaTeX version of the handwritten math shown.",
    "Recreate the math expression in LaTeX based on the image.",
    "From the image, extract the mathematical expression as LaTeX code.",
    "Render this math expression in LaTeX syntax.",
    "What LaTeX markup corresponds to the equation in the picture?",
    "Transcribe this mathematical image into LaTeX.",
    "Provide the LaTeX code that matches this handwritten math formula."
]

'''
In the same chat session as above, we prompt to get the response starter templates using

Now also give 10-15 potential responses that the model can start generating before 
generating the full response.
'''
RESPONSE_TEMPLATES = [
    "Sure, the LaTeX code for the given expression is:",
    "Here's the corresponding LaTeX snippet:",
    "You can write this expression in LaTeX as follows:",
    "The LaTeX representation is:",
    "Below is the LaTeX code that matches the image:",
    "This handwritten expression translates to the following LaTeX:",
    "To reproduce this in LaTeX, use:",
    "Here is the LaTeX markup for the given formula:",
    "This can be expressed in LaTeX like this:",
    "The equivalent LaTeX expression is:",
    "You would write this in LaTeX as:",
    "In LaTeX, this math expression would be:",
    "The formula in LaTeX is:",
    "Sure! It can be written in LaTeX as:",
    "Try the following LaTeX code to generate this expression:"
]


###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

###### Multiprocessing utility

def process_file(args):
    index, (path, split) = args

    try:
        ink = read_inkml_file(path)
        question = random.choice(PROMPTS)
        response_ = ink.annotations.get('normalizedLabel', ink.annotations['label'])
        response = response_ if random.choice([0, 1]) else random.choice(RESPONSE_TEMPLATES) + " {}".format(response_)

        out_path = os.path.join(output_dir, "images", f"{split}_{index}.png")
        display_ink(ink, color='black', fname=out_path)

        image_beginning = random.choice([0, 1])
        if image_beginning:
            convs = [{"from": "human", "value": f"<image>{question}"}, {"from": "gpt", "value": response}]
        else:
            convs = [{"from": "human", "value": f"{question}<image>"}, {"from": "gpt", "value": response}]

        entry = {
            "id": index,
            "image": os.path.join("images", f"{split}_{index}.png"),
            "conversations": convs
        }

        return entry
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return None

###### Class and functions for formatting InkML

@dataclasses.dataclass
class Ink:
  """Represents a single ink, as read from an InkML file."""
  # Every stroke in the ink.
  # Each stroke array has shape (3, number of points), where the first
  # dimensions are (x, y, timestamp), in that order.
  strokes: list[np.ndarray]
  # Metadata present in the InkML.
  annotations: dict[str, str]

def read_inkml_file(filename: str) -> Ink:
  """Simple reader for MathWriting's InkML files."""
  with open(filename, "r") as f:
    root = ElementTree.fromstring(f.read())

  strokes = []
  annotations = {}

  for element in root:
    tag_name = element.tag.replace('{http://www.w3.org/2003/InkML}', '')
    if tag_name == 'annotation':
      annotations[element.attrib.get('type')] = element.text

    elif tag_name == 'trace':
      points = element.text.split(',')
      stroke_x, stroke_y, stroke_t = [], [], []
      for point in points:
        x, y, t = point.split(' ')
        stroke_x.append(float(x))
        stroke_y.append(float(y))
        stroke_t.append(float(t))
      strokes.append(np.array((stroke_x, stroke_y, stroke_t)))

  return Ink(strokes=strokes, annotations=annotations)

def display_ink(
    ink: Ink,
    *,
    figsize: tuple[int, int]=(15, 10),
    linewidth: int=2,
    fname: str="test.png",
    color=None):
  """Simple display for a single ink."""
  pl.figure(figsize=figsize)
  for stroke in ink.strokes:
    pl.plot(stroke[0], stroke[1], linewidth=linewidth, color=color)
  pl.gca().invert_yaxis()
  pl.gca().axis('equal')
  pl.axis('off')
  pl.savefig(fname)
  pl.close()

###### Download the files and unzip

output_dir = "./mathwriting"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

# Download

print('Downloading tar-archive...')
train_files_path = os.path.join(output_dir, "train_files.tgz")
download_file(DOWNLOAD_URL, train_files_path)

# Extract the tar.gz file
print('Extracting folder...')
with tarfile.open(train_files_path, 'r:gz') as tar:
    tar.extractall(output_dir)
    print(f"Extracted images to {output_dir}")

###### Preprocess

all_inkml_files = [(os.path.join(output_dir, "mathwriting-2024/synthetic", x), "synthetic") for x in os.listdir(os.path.join(output_dir, "mathwriting-2024/synthetic")) if x.endswith("inkml")]
all_inkml_files += [(os.path.join(output_dir, "mathwriting-2024/train", x), "train") for x in os.listdir(os.path.join(output_dir, "mathwriting-2024/train")) if x.endswith("inkml")]

with Pool(NUM_WORKERS) as pool:
    results = list(tqdm(pool.imap(process_file, enumerate(all_inkml_files)), total=len(all_inkml_files)))

# Filter out failed entries
new_jsonl_entries = [entry for entry in results if entry is not None]

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'mathwriting/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

os.remove(os.path.join(output_dir, "train_files.tgz"))
shutil.rmtree(os.path.join(output_dir, "mathwriting-2024"))
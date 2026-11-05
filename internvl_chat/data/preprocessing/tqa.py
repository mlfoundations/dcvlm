import os
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import requests
import zipfile
import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
import shutil

TQA_URL = "https://s3.amazonaws.com/ai2-vision-textbook-dataset/dataset_releases/tqa/tqa_train_val_test.zip"
OUTPUT_FILE = "tqa_train_val_test.zip"
EXTRACT_FOLDER = "tqa_dataset"
ROOT = './tqa_dataset/tqa_train_val_test/train'
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
# taken from: https://arxiv.org/pdf/2304.08485 tab. 12
LLAVA_DESC_LONG_PROMPTS = [
    "Describe the following image in detail",
    "Provide a detailed description of the given image",
    "Give an elaborate explanation of the image you see",
    "Share a comprehensive rundown of the presented image",
    "Offer a thorough analysis of the image",
    "Explain the various aspects of the image before you",
    "Clarify the contents of the displayed image with great detail",
    "Characterize the image using a well-detailed description",
    "Break down the elements of the image in a detailed manner",
    "Walk through the important details of the image",
    "Portray the image with a rich, descriptive narrative",
    "Narrate the contents of the image with precision",
    "Analyze the image in a comprehensive and detailed manner",
    "Illustrate the image through a descriptive explanation",
    "Examine the image closely and share its details",
    "Write an exhaustive depiction of the given image",
]
LAB_MAPPING = {i:chr(97+i) for i in range(26)}

######## All utility functions

def count_sentences(caption):
    sentences = sent_tokenize(caption)  # Tokenizes into sentences
    return len(sentences)

def download_file(url, output_file):
    """Downloads a file from the given URL with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_file, "wb") as file, tqdm(
        desc=output_file,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

    print(f"Download complete: {output_file}")

def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")

######## first download and unzip the tqa zip file from: https://s3.amazonaws.com/ai2-vision-textbook-dataset/dataset_releases/tqa/tqa_train_val_test.zip

if not os.path.exists(EXTRACT_FOLDER):
    # Check if the file already exists to avoid re-downloading
    if not os.path.exists(OUTPUT_FILE):
        download_file(TQA_URL, OUTPUT_FILE)
    else:
        print(f"File '{OUTPUT_FILE}' already exists. Skipping download.")

    # Unzip the file and delete the original ZIP
    if os.path.exists(OUTPUT_FILE):
        unzip_file(OUTPUT_FILE, EXTRACT_FOLDER)

######## Move all the images to the tqa/ folder

# first create the output directory
output_dir = 'tqa'
os.makedirs(output_dir, exist_ok=True)

# Define source and destination directories
source_dir = "./tqa_dataset/tqa_train_val_test/train"
destination_dir = "./tqa/images"

# List of folders to move
folders_to_move = ["abc_question_images", "question_images", "teaching_images", "textbook_images"]

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Move each folder
for folder in folders_to_move:
    src_path = os.path.join(source_dir, folder)
    dest_path = os.path.join(destination_dir, folder)

    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"Moved: {src_path} -> {dest_path}")
    else:
        print(f"Skipping move: {src_path} (Does not exist)")

######## We now want to start the reformatting procedure

# load the json file
with open(os.path.join(ROOT, 'tqa_v1_train.json')) as f:
    json_contents = json.load(f)

# Next start extracting all the potential image-caption pairs possible
print("Extracting all potential image-caption pairs!")

image_caption_pairs = []

for lesson in json_contents:
    
    # First extract from instructionalDiagrams
    if 'instructionalDiagrams' in lesson:
        if len(lesson['instructionalDiagrams']) > 0:
            for k, v in lesson['instructionalDiagrams'].items():
                if 'imagePath' in v and 'processedText' in v:
                    image_caption_pairs.append([v['imagePath'], v['processedText']])

    # Second extract from adjunctTopics
    if 'adjunctTopics' in lesson:
        for subsec in lesson['adjunctTopics'].values():
            if 'content' in subsec and 'figures' in subsec['content']:
                for d in subsec['content']['figures']:
                    image_caption_pairs.append([d['imagePath'], d['caption']])

    # Third extract from topics
    if 'topics' in lesson:
        for subsec in lesson['topics'].values():
            if 'content' in subsec and 'figures' in subsec['content']:
                for d in subsec['content']['figures']:
                    image_caption_pairs.append([d['imagePath'], d['caption']])

figure_pattern = re.compile(r'^FIGURE \d+\.\d+$')
figure_remove_pattern = re.compile(r'^\d*\s*FIGURE\s+\d+\.\d+\s*')

image_caption_jsonl_entries = []

for index, imgcap in enumerate(image_caption_pairs):

    # ensure that we filter out captions that contain only FIGURE X.X and nothing else
    if figure_pattern.match(imgcap[1]):
        continue

    # is it a long or a short caption?
    num_sents = count_sentences(imgcap[1])

    # pick the prompt
    if num_sents == 1: # if its exactly one sentence then add brief prompt
        prompt_ = random.choice(LLAVA_DESC_SHORT_PROMPTS)
    else:
        prompt_ = random.choice(LLAVA_DESC_LONG_PROMPTS) 

    # preprocess the caption to remove FIGURE X.X at the beginning of some captions
    processed_cap = re.sub(figure_remove_pattern, '', imgcap[1]).strip()

    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])

    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":processed_cap}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":processed_cap}]

    entry_ = {"id":len(image_caption_jsonl_entries), "image": 'images/'+imgcap[0], "conversations": convs}
    image_caption_jsonl_entries.append(entry_)

with open('./tqa/image_caption_preprocessed.jsonl', 'w') as f:
    for entry in image_caption_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing image-captions complete! Saved as 'tqa/image_caption_preprocessed.jsonl' with {} total samples.".format(len(image_caption_jsonl_entries)))

# Next start extracting all the potential non-image qa pairs possible
print("Extracting all potential non-image qa pairs!")

non_image_qas = []

for lesson in json_contents:
    
    # extract the pool of such questions
    non_image_pool = lesson['questions']['nonDiagramQuestions']

    # get all qs in pool
    for qdict in non_image_pool.values():
        if 'beingAsked' in qdict and 'answerChoices' in qdict and 'correctAnswer' in qdict and 'questionType' in qdict:
            # extract question
            q_ = qdict['beingAsked']['processedText']
            # extract correct answer
            a_ = qdict['correctAnswer']['processedText']
            ra_ = qdict['correctAnswer']['rawText']
            # extract answer choices
            cs_ = [x['rawText'] for x in list(qdict['answerChoices'].values())]

            ##### NOTE: here i am using the format specified for mcq evals in vlmevalkit, is this contamination???
            ##### see mcq eval format here: https://github.com/open-compass/VLMEvalKit/blob/2d8e44e465bb89ff55ffaf54f989449d96c32af5/vlmeval/dataset/image_mcq.py#L180-L190
            prompt_ = 'Question: {}\n'.format(q_)
            for o in cs_:
                prompt_ += '{}\n'.format(o)
            prompt_ += 'Please select the correct answer from the options above. \n'
            try:
                answer_ = a_ + '. ' + qdict['answerChoices'][a_]['processedText']
            except KeyError as e: # sometimes there's weird formatting errors, skip them!
                continue

            non_image_qas.append([prompt_, answer_])

text_only_jsonl_entries = []

for q_, a_ in non_image_qas:
    convs = [{"from":"human", "value":"{}".format(q_)}, {"from":"gpt", "value":a_}]
    entry_ = {"id":len(text_only_jsonl_entries), "conversations": convs}
    text_only_jsonl_entries.append(entry_)

with open('./tqa/text_only_qa_preprocessed.jsonl', 'w') as f:
    for entry in text_only_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing text-only qas complete! Saved as 'tqa/text_only_qa_preprocessed.jsonl' with {} total samples.".format(len(text_only_jsonl_entries)))

# Next start extracting all the potential image-based qa pairs possible
print("Extracting all potential image-based qa pairs!")

image_qas = []

for lesson in json_contents:
    
    # extract the pool of such questions
    image_pool = lesson['questions']['diagramQuestions']

    # get all qs in pool
    for qdict in image_pool.values():
        if 'beingAsked' in qdict and 'answerChoices' in qdict and 'correctAnswer' in qdict and 'questionType' in qdict and 'imagePath' in qdict:
            # extract question
            q_ = qdict['beingAsked']['processedText']
            # extract correct answer
            a_ = qdict['correctAnswer']['processedText']
            ra_ = qdict['correctAnswer']['rawText']
            # extract answer choices
            cs_ = [x['rawText'] for x in list(qdict['answerChoices'].values())]
            # extract image path
            img_ = qdict['imagePath']
            assert isinstance(img_, str) and img_.endswith('.png'), 'Error loading image: {}'.format(img_)

            ##### NOTE: here i am using the format specified for mcq evals in vlmevalkit, is this contamination???
            ##### see mcq eval format here: https://github.com/open-compass/VLMEvalKit/blob/2d8e44e465bb89ff55ffaf54f989449d96c32af5/vlmeval/dataset/image_mcq.py#L180-L190
            prompt_ = 'Question: {}\n'.format(q_)
            for ind, o in enumerate(cs_):
                if not o.startswith(LAB_MAPPING[ind]):
                    o = LAB_MAPPING[ind] + '. ' + o
                prompt_ += '{}\n'.format(o)
            prompt_ += 'Please select the correct answer from the options above. \n'
            try:
                answer_ = a_ + '. ' + qdict['answerChoices'][a_]['processedText']
            except KeyError as e: # sometimes there's weird formatting errors, skip them!
                continue

            image_qas.append([img_, prompt_, answer_])

image_jsonl_entries = []

for img_, q_, a_ in image_qas:

    #### NOTE: should we do this here?
    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(q_)}, {"from":"gpt", "value":a_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(q_)}, {"from":"gpt", "value":a_}]

    entry_ = {"id":len(image_jsonl_entries), "image": 'images/'+img_, "conversations": convs}
    image_jsonl_entries.append(entry_)

with open('./tqa/image_qa_preprocessed.jsonl', 'w') as f:
    for entry in image_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing image qas complete! Saved as 'tqa/image_qa_preprocessed.jsonl' with {} total samples.".format(len(text_only_jsonl_entries)))

shutil.rmtree('tqa_dataset')
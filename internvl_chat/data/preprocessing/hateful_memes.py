import kagglehub
import os
import shutil
import json
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl


'''
Sourced from gpt-4O from the chatgpt-UI with the following prompt:

I have a dataset of memes, images with interleaved humorous captions on top. I also have the corresponding texts for the humorous captions. For each of these images and corresponding texts, the task is to perform hate speech detection, to classify whether the corresponding caption is hateful or contains hate speech. I want to convert this dataset into an instruction tuning dataset where the prompt would be something like "Is this statement coupled with the image hateful?", where a positive response would imply hateful content. Please provide a list of 15 different such prompts I can use as part of the visual instruction tuning dataset that I can randomly sample when constructing the dataset.
'''

PROMPTS_LIST = [
        'Does this meme contain hateful content when considering both the image and text?',
        'Is the message conveyed by this meme offensive or hateful?',
        'Would you classify the combination of this image and text as hate speech?',
        'Does the caption, in the context of the image, promote hate or discrimination?',
        'Analyze this meme: Does it include any hateful or offensive language?',
        'Is there any hateful intent in the way this text and image are combined?',
        'Would this meme be considered harmful or hateful to a particular group?',
        'Assess this meme: Does the humor rely on hateful or offensive themes?',
        'Does this image-text combination spread hate or reinforce harmful stereotypes?',
        'Looking at both the image and caption, does this meme contain hate speech?',
        'Would this meme reasonably be considered offensive or hateful?',
        'Does this meme ridicule or demean a group in a hateful way?',
        'Does the text, when paired with this image, contribute to hate speech?',
        'Would an objective viewer find this meme hateful or discriminatory?',
        'Is this meme an example of hate speech or offensive content?',
        ]



# Set Kaggle credentials at runtime
assert 'KAGGLE_USERNAME' in os.environ, 'Please set-up your KAGGLE_USERNAME env variable for your kaggle username, set is using export KAGGLE_USERNAME=XXXXX'
assert 'KAGGLE_KEY' in os.environ, 'Please set-up your KAGGLE_KEY env variable for your kaggle key, set is using export KAGGLE_KEY=XXXXX'

ROOT_DIR = './hateful_memes'
IMAGE_DIR = ROOT_DIR + '/images'
HM_TRAIN_JSONL = 'data/train.jsonl'

os.makedirs(ROOT_DIR, exist_ok=True)

# Need to set-up the cache directory to prevent Kaggle from downloading it to ~/.cache
os.environ["KAGGLEHUB_CACHE"] = ROOT_DIR

# Download latest version
kaggle_path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset")

# Images are saved under: datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/img
print(f"Dataset saved in {kaggle_path} by Kaggle...")

# Copy images from the Kaggle-saved directory to the IMAGE_DIR
print(f"Copying the images from Kaggle-saved directory to {IMAGE_DIR}...")
shutil.move(os.path.join(kaggle_path, 'data/img/'), IMAGE_DIR)

# Load the original jsonl file
with open(os.path.join(kaggle_path, HM_TRAIN_JSONL), 'r') as f:
    original_train_annotations = [json.loads(line) for line in f.readlines()]

new_jsonl_entries = []

for index, instance in tqdm(enumerate(original_train_annotations), ascii=True, total=len(original_train_annotations)):
    pre_prompt = random.choice(PROMPTS_LIST)
    gt_caption = instance['text']
    
    prompt_ = "{} {}".format(pre_prompt, gt_caption)
    answer_ = "True" if instance['label'] == 1 else "False"

    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    if image_beginning:
        convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    else:
        convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":answer_}]

    image_path = instance['img'][4:]
    assert os.path.exists(os.path.join(IMAGE_DIR, image_path)), 'Image {} does not exist!!!'.format(image_path)

    entry_ = {"id": len(new_jsonl_entries), "image": 'images/'+ image_path, "conversations": convs}
    new_jsonl_entries.append(entry_)


with open('./hateful_memes/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(ROOT_DIR, 'datasets'))

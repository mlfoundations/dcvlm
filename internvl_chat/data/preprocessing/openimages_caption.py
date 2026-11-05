import fiftyone as fo
import fiftyone.zoo as foz
import os
import json
import shutil
from tqdm import tqdm
import requests
import random
random.seed(0)

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

# We hardcode this to the path where we want to write our data to
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = "./openimages_caption"
fo.config.dataset_zoo_dir = "./openimages_caption"

assert 'FIFTYONE_DATASET_ZOO_DIR' in os.environ, 'Please set-up your FIFTYONE_DATASET_ZOO_DIR env variable for your fiftyone dataset directory (preferably ./openimages_caption), using export FIFTYONE_DATASET_ZOO_DIR=XXXXX'

ROOT_DIR = './openimages_caption'
IMAGE_DIR = ROOT_DIR + '/images'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Returns the links for each shard of the train annotations
def get_shard_link(shard_idx):
    return "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-0000"+shard_idx+"-of-00010.jsonl"

# Downloads the given url, names it based on the output file
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


# Only chose the detection label version, otherwise it downloads every single label for every single type which can take >10 hours.
dataset = foz.download_zoo_dataset(
        "open-images-v6",
        split="train",
        label_types=["detections"])

# Images are saved under the following directory
saved_data_dir = os.path.join(ROOT_DIR, 'open-images-v6/train/data/')

# Copy images from the Kaggle-saved directory to the IMAGE_DIR
print(f"Copying the images from fiftyone-saved directory to {IMAGE_DIR}...")
shutil.copytree(saved_data_dir, IMAGE_DIR, copy_function=shutil.move, dirs_exist_ok=True)

# Also obtain the original captions...
openimages_train_annotations = []
for idx in tqdm(range(0, 10)):
    shard_url = get_shard_link(str(idx))
    file_name = "train_"+str(idx)+".jsonl"
    if not os.path.exists(os.path.join(ROOT_DIR, file_name)):
        download_file(shard_url, os.path.join(ROOT_DIR, file_name))

    with open(os.path.join(ROOT_DIR, file_name), 'r') as f:
        jsonl_contents = [json.loads(line) for line in f.readlines()]
        
        # Remove all of the keys except the image ids and captions before storing...
        # Otherwise the annotations could be too heavy on the CPU memory
        for sample in jsonl_contents:
            keys_to_delete = list(sample.keys())
            keys_to_delete.remove('caption')
            keys_to_delete.remove('image_id')
            
            for key in keys_to_delete:
                del sample[key]
        
        openimages_train_annotations += jsonl_contents

# Start reformatting the annotations
new_jsonl_entries = []
annotated_images = []

for index, instance in tqdm(enumerate(openimages_train_annotations), ascii=True, total=len(openimages_train_annotations)):
    
    image_path = instance['image_id'].zfill(16) + '.jpg'

    assert os.path.exists(os.path.join(IMAGE_DIR, image_path)), 'Image {} does not exist!!!'.format(image_path)
    
    # select a prompt from one of the LLaVA short caption descriptions
    prompt_ = random.choice(LLAVA_DESC_SHORT_PROMPTS)

    # response should simply be the ground truth caption
    response_ = instance['caption']
    
    # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    image_beginning = random.choice([0, 1])
    if image_beginning:
        convs = [{"from":"human", "value":"<image>\n{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}\n<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    
    entry_ = {"id": len(new_jsonl_entries), "image": "images/"+image_path, "conversations": convs}
    new_jsonl_entries.append(entry_)
    annotated_images.append(image_path)


# Remove the images without the caption annotations
all_images = os.listdir(IMAGE_DIR)
removed_cnt = 0
for idx, img_path in tqdm(enumerate(all_images), ascii=True, total=len(all_images)):

    # If the relevant image does not have captions, remove
    if img_path not in annotated_images:
        os.remove(os.path.join(IMAGE_DIR, img_path))
        removed_cnt += 1

# Final check for whether all train images are still remaining
for index, instance in tqdm(enumerate(openimages_train_annotations), ascii=True, total=len(openimages_train_annotations)):
    image_path = instance['image_id'].zfill(16) + '.jpg'
    assert os.path.exists(os.path.join(IMAGE_DIR, image_path)), 'Image {} does not exist!!!'.format(image_path)

print(f'{removed_cnt} images without captions are removed, only {len(new_jsonl_entries)} images are remaining...')

with open(os.path.join(ROOT_DIR, 'preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'openimages_caption/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

shutil.rmtree(os.path.join(ROOT_DIR, 'open-images-v6'))
os.remove(os.path.join(ROOT_DIR, 'train_0.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_1.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_2.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_3.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_4.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_5.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_6.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_7.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_8.jsonl'))
os.remove(os.path.join(ROOT_DIR, 'train_9.jsonl'))

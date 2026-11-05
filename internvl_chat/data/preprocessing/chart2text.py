import os
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import requests
import git
from tqdm import tqdm

'''
Sourced from gpt-4o from the chatgpt-UI with the following prompt:

I have a dataset of images of different charts sourced from statista. I have corresponding caption descriptions of what the chart contains and titles of each of the charts. I want to convert this dataset into an instruction tuning dataset where the prompt would be something like describe the chart image or what are some salient aspects in the chart.
Please provide a list of 15 different such prompts I can use as part of the visual instruction tuning dataset that I can randomly sample when constructing the dataset.

'''
PROMPTS_LIST = [
    "Describe the key insights from this chart.",
    "Summarize the main trends and patterns visible in this chart.",
    "What are the most important takeaways from this chart?",
    "Explain the significance of the data presented in this chart.",
    "Identify and describe the major changes or variations in this chart.",
    "What are some notable outliers or unusual trends in this chart?",
    "Provide a detailed analysis of the information shown in this chart.",
    "What does this chart reveal about the topic it represents?",
    "Break down the different elements of this chart and their meaning.",
    "How do the data points in this chart compare to each other?",
    "Highlight any interesting correlations or relationships in this chart.",
    "Interpret the data in this chart and explain its implications.",
    "If you had to summarize this chart in one sentence, what would it be?",
    "How does this chart visually represent the underlying dataset?",
    "What story does this chart tell, and what conclusions can be drawn?"
]

output_dir = "./chart2text"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

# Define repository and folder details
repo_url = "https://github.com/JasonObeid/Chart2Text.git"
image_repo_url = "https://github.com/JasonObeid/Chart2TextImages.git"
clone_dir = "Chart2Text_repo"
clone_dir_images = "Chart2TextImages_repo"
keep_folders = ["dataset/captions", "dataset/titles", "dataset/multiColumn/captions", "dataset/multiColumn/titles"]
keep_folders_images = ["images", "multiColumn/images"]
destination = "chart2text/tempdata"

# Clone the repository
print("Cloning Chart2Text repository...")
git.Repo.clone_from(repo_url, clone_dir, depth=1)

# Construct paths
keep_paths = [os.path.join(clone_dir, keep_folder) for keep_folder in keep_folders]

# Ensure the folder exists
for keep_folder, keep_path in zip(keep_folders, keep_paths):
    if os.path.exists(keep_path):
        # Move the desired folder to the new location
        print(f"Moving {keep_folder} to {destination}...")
        shutil.move(keep_path, os.path.join(destination, keep_folder))
    else:
        print(f"Folder {keep_folder} not found in the repository!")

# Delete the cloned repository
shutil.rmtree(clone_dir)
print("Cleanup completed. Only relevant folders are retained in 'chart2text/tempdata'.")

# Clone the image repository
print("Cloning Chart2TextImages repository...")
git.Repo.clone_from(image_repo_url, clone_dir_images, depth=1)

# Construct paths
keep_paths_images = [os.path.join(clone_dir_images, keep_folder) for keep_folder in keep_folders_images]

# Ensure the folder exists
for keep_folder, keep_path in zip(keep_folders_images, keep_paths_images):
    if os.path.exists(keep_path):
        # Move the desired folder to the new location
        print(f"Moving {keep_folder} to {destination}...")
        shutil.move(keep_path, os.path.join(destination, keep_folder))
    else:
        print(f"Folder {keep_folder} not found in the repository!")

# Delete the cloned repository
shutil.rmtree(clone_dir_images)
print("Cleanup completed. Only relevant folders are retained in 'chart2text/tempdata'.")

# Define source and destination directories
source_dirs = ["./chart2text/tempdata/images", "./chart2text/tempdata/multiColumn"]
destination_dir = "./chart2text/images"

# Move each folder while preserving subfolder structure
for source_dir in source_dirs:
    dest_subdir = os.path.join(destination_dir, os.path.basename(source_dir))  # Keep folder name
    if os.path.exists(source_dir):
        # Remove destination if it exists to prevent conflicts
        if os.path.exists(dest_subdir):
            shutil.rmtree(dest_subdir)

        # Move the entire folder tree
        shutil.move(source_dir, dest_subdir)
        print(f"Moved: {source_dir} -> {dest_subdir}")
    else:
        print(f"Skipping move: {source_dir} (Does not exist)")

# NOTE: we dont use the matplot images, but only use the statista images since they are the original ones
if os.path.exists('./chart2text/images/images/matplot'):
    shutil.rmtree('./chart2text/images/images/matplot')
if os.path.exists('./chart2text/images/multiColumn/images/matplot'):
    shutil.rmtree('./chart2text/images/multiColumn/images/matplot')
print('Removed unnecessary files!')

new_jsonl_entries = []

##### First, read and create json contents for single column tables

single_col_caps_path = './chart2text/tempdata/dataset/captions'
single_col_titles_path = './chart2text/tempdata/dataset/titles'

captions = {}
titles = {}

for txt_file in os.listdir(single_col_caps_path):
    with open(os.path.join(single_col_caps_path, txt_file), 'r') as f:
        cap = f.read()
        captions[txt_file] = cap

for txt_file in os.listdir(single_col_titles_path):
    with open(os.path.join(single_col_titles_path, txt_file), 'r') as f:
        title = f.read()
        titles[txt_file] = title

images_available_statista = os.listdir('./chart2text/images/images/statista')

for key in tqdm(captions, ascii=True, total=len(captions)):

    cap_ = captions[key]
    title_ = titles[key]

    assert key in titles, 'caption key not available in titles!'
    assert key.split('.')[0]+'.png' in images_available_statista, 'caption key not available in statista-images!'

    prompt_ = random.choice(PROMPTS_LIST)
    # randomly pick if title comes at the beginning or at the end of the response, this is to prevent overfitting to one style of response
    title_in_beginning = random.choice([0, 1])
    if title_in_beginning:
        response_ = 'This chart is titled "{}". {}'.format(title_, cap_)
    else:
        response_ = '{}. This chart is titled "{}".'.format(cap_, title_)

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    entry_ = {"id":len(new_jsonl_entries), "image": 'images/images/statista/'+'{}.png'.format(key.split('.')[0]), "conversations": convs}
    new_jsonl_entries.append(entry_)

##### Second, read and create json contents for multi column tables

mul_col_caps_path = './chart2text/tempdata/dataset/multiColumn/captions'
mul_col_titles_path = './chart2text/tempdata/dataset/multiColumn/titles'

captions = {}
titles = {}

for txt_file in os.listdir(mul_col_caps_path):
    with open(os.path.join(mul_col_caps_path, txt_file), 'r') as f:
        cap = f.read()
        captions[txt_file] = cap

for txt_file in os.listdir(mul_col_titles_path):
    with open(os.path.join(mul_col_titles_path, txt_file), 'r') as f:
        title = f.read()
        titles[txt_file] = title

images_available_statista = os.listdir('./chart2text/images/multiColumn/images/statista')

for key in tqdm(captions, ascii=True, total=len(captions)):

    cap_ = captions[key]
    title_ = titles[key]

    assert key in titles, 'caption key not available in titles!'
    assert key.split('.')[0]+'.png' in images_available_statista, 'caption key not available in statista-images!'

    prompt_ = random.choice(PROMPTS_LIST)
    # randomly pick if title comes at the beginning or at the end of the response, this is to prevent overfitting to one style of response
    title_in_beginning = random.choice([0, 1])
    if title_in_beginning:
        response_ = 'This chart is titled "{}". {}'.format(title_, cap_)
    else:
        response_ = '{}. This chart is titled "{}".'.format(cap_, title_)

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":response_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":response_}]
    entry_ = {"id":len(new_jsonl_entries), "image": 'images/multiColumn/images/statista/'+'{}.png'.format(key.split('.')[0]), "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./chart2text/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'chart2text/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

shutil.rmtree(os.path.join(output_dir, 'tempdata'))
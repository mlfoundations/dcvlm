import os
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import requests
import git

LAB_MAPPING = {i:chr(97+i) for i in range(26)}

output_dir = "./tabmwp"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

# Define repository and folder details
repo_url = "https://github.com/lupantech/PromptPG.git"
clone_dir = "PromptPG_repo"
keep_folder = "data/tabmwp"
destination = "tabmwp/tempdata"

# Clone the repository
print("Cloning TabMWP repository...")
git.Repo.clone_from(repo_url, clone_dir, depth=1)

# Construct paths
keep_path = os.path.join(clone_dir, keep_folder)

# Ensure the folder exists
if os.path.exists(keep_path):
    # Move the desired folder to the new location
    print(f"Moving {keep_folder} to {destination}...")
    shutil.move(keep_path, destination)
else:
    print(f"Folder {keep_folder} not found in the repository!")

# Delete the cloned repository
shutil.rmtree(clone_dir)
print("Cleanup completed. Only 'tabmwp' folder is retained in 'tabmwp/tempdata'.")

# Move the table images to the right directory
# Define source and destination directories
source_dir = "./tabmwp/tempdata/tables"
destination_dir = "./tabmwp/images"

# Move each folder
if os.path.exists(source_dir):
    shutil.move(source_dir, destination_dir)
    print(f"Moved: {source_dir} -> {destination_dir}")
else:
    print(f"Skipping move: {source_dir} (Does not exist)")

# Load train json
with open('./tabmwp/tempdata/problems_train.json') as f:
    json_contents = json.load(f)

new_jsonl_entries = []

for cnt, (key, d) in enumerate(json_contents.items()):

    question = d['question']
    choices = d['choices']
    answer = d['answer']
    unit = d['unit']
    solution = d['solution']
    qtype = d['ques_type']
    atype = d['ans_type']

    assert os.path.exists(os.path.join(destination_dir, 'tables', '{}.png'.format(key))), 'Image {} does not exist!!!'.format(key)

    prompt_ = ''
    assert question is not None, 'question is None!'
    prompt_ += 'Question: {}\n'.format(question)
    if qtype == 'multi_choice':
        assert choices is not None, 'question is mcq style, but choices is None!!!'
        prompt_ += 'Options:\n'
        for indd, o in enumerate(choices):
            prompt_ += '{}. {}\n'.format(LAB_MAPPING[indd], o)

    if solution is None:
        answer_ = 'Final Answer: {}'.format(answer)
    else:    
        answer_ = '{}\nFinal Answer: {}'.format(solution, answer)
    if unit is not None:
        answer_ += ' {}'.format(unit)

    image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
    if image_beginning:
        convs = [{"from":"human", "value":"<image>{}".format(prompt_)}, {"from":"gpt", "value":answer_}]
    else:
        convs = [{"from":"human", "value":"{}<image>".format(prompt_)}, {"from":"gpt", "value":answer_}]
    entry_ = {"id":cnt, "image": 'images/tables/'+'{}.png'.format(key), "conversations": convs}
    new_jsonl_entries.append(entry_)

with open('./tabmwp/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

print("Preprocessing complete! Saved as 'tabmwp/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))

shutil.rmtree(os.path.join(output_dir, 'tempdata'))
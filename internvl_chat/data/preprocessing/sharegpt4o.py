from huggingface_hub import snapshot_download
import os
import zipfile
import json
import shutil
from tqdm import tqdm

#### NOTE: you have to first enter your details on the HF dataset page since this is a gated access dataset. Once you sign up and enter your details, you should be able to login.

if 'HF_TOKEN' not in os.environ:
    raise ValueError('Please set your HF_TOKEN environment variable. If you dont have one, create one at: https://huggingface.co/settings/tokens and do export HF_TOKEN=hf_XXXXXX.')

output_dir = './sharegpt4o'
os.makedirs(output_dir, exist_ok=True)
img_path_prefix = 'mnt/petrelfs/wangwenhai/workspace_cef/4o/image/'

snapshot_download(
    repo_id="OpenGVLab/ShareGPT-4o",
    repo_type="dataset",
    allow_patterns=["images.zip"],
    local_dir=output_dir,
    token=os.environ['HF_TOKEN']
)

snapshot_download(
    repo_id="OpenGVLab/ShareGPT-4o",
    repo_type="dataset",
    allow_patterns=["image_conversations/gpt-4o.jsonl"],
    local_dir=output_dir,
    token=os.environ['HF_TOKEN']
)

print("Extracting all images from zip file")
with zipfile.ZipFile(os.path.join(output_dir, 'images.zip'), 'r') as zip_ref:
    zip_ref.extractall(output_dir)

out_data = []
with open(os.path.join(output_dir, 'image_conversations/gpt-4o.jsonl'), 'r') as f:
    all_data = list(f.readlines())
    for idx in tqdm(range(len(all_data))):
        datum = json.loads(all_data[idx])
        datum['image'] = os.path.join(img_path_prefix, datum['image'])
        datum.update({'id': idx})
        out_data.append(datum)

with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as fout:
    for datum in out_data:
        fout.write(json.dumps(datum) + "\n")

os.remove(os.path.join(output_dir, 'images.zip'))
shutil.rmtree(os.path.join(output_dir, 'image_conversations'))
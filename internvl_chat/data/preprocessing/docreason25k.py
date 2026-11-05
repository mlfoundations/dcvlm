import os
from huggingface_hub import snapshot_download
import tarfile
from tqdm import tqdm
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

output_dir = "./docreason25k"
os.makedirs(output_dir, exist_ok=True)

snapshot_download(
    repo_id="mPLUG/DocReason25K",
    repo_type="dataset",
    allow_patterns=["detailed_explanation.jsonl", "imgs.tar.gz"],
    local_dir='.'
)

with tarfile.open('imgs.tar.gz', 'r:gz') as tar:
    tar.extractall(output_dir)

out_data = []
with open('detailed_explanation.jsonl', 'r') as f:
    all_data = list(f.readlines())
    for idx in tqdm(range(len(all_data))):
        datum = json.loads(all_data[idx])
        assert len(datum['image']) == 1
        new_datum = {}
        new_datum['id'] = len(out_data)
        new_datum['image'] = datum['image'][0].replace('./', '')
        im = Image.open(os.path.join(output_dir, new_datum['image']))
        width, height = im.size
        new_datum.update({'id': len(out_data), 'width': width, 'height': height})

        conversations = []
        for turn in datum['messages']:
            if turn['role'] == 'user':
                conversations.append({"from": "human", "value": turn["content"].replace('<|image|>', '<image>')})
            else:
                assert '<|image|>' not in turn['content']
                conversations.append({"from": "gpt", "value": turn["content"]})
        new_datum['conversations'] = conversations

        out_data.append(new_datum)
        
with open(os.path.join(output_dir, 'preprocessed.jsonl'), 'w') as fout:
    for datum in out_data:
        fout.write(json.dumps(datum) + "\n")

os.remove('detailed_explanation.jsonl')
os.remove('imgs.tar.gz')

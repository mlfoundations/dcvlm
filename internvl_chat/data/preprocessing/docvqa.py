import wget
import os
import json
from tqdm import tqdm
from PIL import Image
import tarfile

wget.download('https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/docvqa.tar.gz')
with tarfile.open('docvqa.tar.gz', 'r:gz') as tar:
    tar.extractall('.')
assert os.path.isdir('./docvqa/documents/')

if not os.path.exists('Cambrian10M.jsonl'):
    wget.download('https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/resolve/main/jsons/Cambrian10M.jsonl')

cnt = 0
out_data = []
with open('Cambrian10M.jsonl', 'r') as f:
    all_data = list(f.readlines())
    for idx in tqdm(range(len(all_data))):
        datum = json.loads(all_data[idx])
        if 'image' not in datum:
            continue
        if 'docvqa' in datum['image']:
            image_filename = datum['image'].split('/')[-1]
            if not os.path.exists(f'./docvqa/documents/{image_filename}'):
                continue
            im = Image.open(f'./docvqa/documents/{image_filename}')
            width, height = im.size
            datum.update({'image': f'documents/{image_filename}', 'width': width, 'height': height, 'id': cnt})
            datum.pop('source', None)
            out_data.append(datum)
            cnt += 1

with open('./docvqa/preprocessed.jsonl', 'w') as f:
    for entry in out_data:
        f.write(json.dumps(entry) + '\n')

imgs_saved = len(os.listdir('./docvqa/documents'))
print("Preprocessing complete! Saved as 'preprocessed.jsonl' with {} saved images and {} total samples.".format(imgs_saved, cnt+1))
os.remove('docvqa/images/')
os.remove('docvqa.tar.gz')
os.remove('Cambrian10M.jsonl')
from huggingface_hub import snapshot_download
import os
import json
from tqdm import tqdm
import zipfile
from pathlib import Path
import gdown
import wget

output_dir = 'sharegpt4v'
os.makedirs(output_dir, exist_ok=True)

# Download LAION-CC-SBU-558K
print("extracting llava pretrain data")
Path("./sharegpt4v/data/llava/llava_pretrain/images/").mkdir(parents=True, exist_ok=True)
snapshot_download(
  repo_id="liuhaotian/LLaVA-Pretrain",
  repo_type="dataset",
  allow_patterns=['images.zip'],
  local_dir="sharegpt4v"
)
with zipfile.ZipFile('./sharegpt4v/images.zip') as zfp:
    zfp.extractall('./sharegpt4v/data/llava/llava_pretrain/images/')

# Download COCO data
assert 'COCO_DIR' in os.environ, 'Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'

# Download WebData
IMAGE_URL = [
             'https://drive.google.com/uc?id=1f4v_3e1OJtyYqam1CEp6RenCNTU5_mG2',
             'https://drive.google.com/uc?id=1JpJkN7ZMA50xAhMx9O-rVb5yLhfGm3_o',
             'https://drive.google.com/uc?id=1FxB2Nw-vWUcTUSI_dBpPIykb-uGYoEqV',
             'https://drive.google.com/uc?id=1-SB71C3j1mVg0kDDXwj2IWGEoBoRUD-J',
             ]
IMAGE_ZIP = ['share_textvqa.zip', 'web-landmark.zip', 'wikiart.zip', 'web-celebrity.zip']
for image_url, image_zip in zip(IMAGE_URL, IMAGE_ZIP):
    gdown.download(image_url, os.path.join(output_dir, image_zip))
    folder_name = image_zip.split('.zip')[0]
    print(f"extracting {folder_name} data")
    #Path(f"./sharegpt4v/{folder_name}/images/").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(os.path.join(output_dir, image_zip)) as zfp:
        zfp.extractall(f'./sharegpt4v/')

#Download Visual Genomes
print("extracting Visual Genomes data")
IMAGE_URL = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"]
IMAGE_ZIP = ["VG_100K.zip", "VG_100K_2.zip"]
for image_url, image_zip in zip(IMAGE_URL, IMAGE_ZIP):
    wget.download(image_url, out=os.path.join(output_dir, image_zip))
    folder_name = image_zip.split('.zip')[0]
    Path("./sharegpt4v/data/vg/").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(os.path.join(output_dir, image_zip)) as zfp:
        zfp.extractall('./sharegpt4v/data/vg/')

assert os.path.exists('./gqa')
assert os.path.exists('./ocr_vqa')
assert os.path.exists('./textvqa')

snapshot_download(
  repo_id="Lin-Chen/ShareGPT4V",
  repo_type="dataset",
  allow_patterns=["share-captioner_coco_lcs_sam_1246k_1107.json", "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"],
  local_dir="sharegpt4v"
)


def preprocess_json(in_file, out_file, skip_sam=True):
    with open(in_file, 'r') as fin:
        json_contents = json.load(fin)

    preprocessed = []
    for index, json_contents in tqdm(enumerate(json_contents), ascii=True, total=len(json_contents)):
        convs = json_contents['conversations']

        if 'image' not in json_contents:
            entry_ = {"id": len(preprocessed), "conversations": convs}
            preprocessed.append(entry_)
            continue
        image_path = json_contents['image']
        if skip_sam and 'sam/' in image_path:
            continue

        assert convs is not None, 'conversations is None!'
        assert image_path is not None, 'image_path is None!'
        if any([dataset_name in image_path for dataset_name in ['coco', 'gqa', 'ocr_vqa', 'textvqa']]):
            old_image_path = image_path
            image_path = '/'.join(old_image_path.split('/')[1:])
            if 'coco' in old_image_path:
                split_ = image_path.split("/")[0]
                image_path = os.path.join('images', split_, image_path)
                assert os.path.exists(os.path.join(os.environ["COCO_DIR"], image_path)), image_path
        else:
            image_path = os.path.join('data', image_path)
            assert os.path.exists(os.path.join(output_dir, image_path)), image_path

        entry_ = {"id": len(preprocessed), "image": image_path, "conversations": convs}
        preprocessed.append(entry_)

    with open(out_file, 'w') as fout:
        for entry in preprocessed:
            fout.write(json.dumps(entry) + '\n')

    print("Preprocessing complete! Saved as '{}' with {} total samples.".format(out_file, len(preprocessed)))
    os.remove(in_file)


preprocess_json('./sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json', './sharegpt4v/finetune.jsonl', skip_sam=True)
preprocess_json('./sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json', './sharegpt4v/pretrain_1246k.jsonl', skip_sam=True)

for filename in os.listdir(output_dir):
    if filename.endswith('.zip'):
        os.remove(os.path.join(output_dir, filename))

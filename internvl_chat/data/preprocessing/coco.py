import os
import json
import requests
import shutil
import zipfile
import io

COCO_2014_TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2014.zip'
COCO_2014_VAL_IMAGES_URL = 'http://images.cocodataset.org/zips/val2014.zip'
COCO_2014_TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/test2014.zip'
COCO_2017_TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip'
COCO_2017_VAL_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'
COCO_2017_TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/test2017.zip'

output_dir = './coco'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+'/images', exist_ok=True)

###### Utility functions

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

# Function to unzip files
def unzip_file(zip_path, extract_path):
    """Extracts a ZIP file and removes it after extraction."""
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete: {extract_path}")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Deleted ZIP file: {zip_path}")

###### Download and unzip all datasets

# coco-2014-train
print('Downloading: {}'.format(COCO_2014_TRAIN_IMAGES_URL))
dpath_ = os.path.join(output_dir, 'train2014.zip')
download_file(COCO_2014_TRAIN_IMAGES_URL, dpath_)
upath_ = os.path.join(output_dir, 'images', 'train2014')
unzip_file(dpath_, upath_)
print('Cleaned up and ready in: {}!'.format(upath_))

# coco-2014-val
print('Downloading: {}'.format(COCO_2014_VAL_IMAGES_URL))
dpath_ = os.path.join(output_dir, 'val2014.zip')
download_file(COCO_2014_VAL_IMAGES_URL, dpath_)
upath_ = os.path.join(output_dir, 'images', 'val2014')
unzip_file(dpath_, upath_)
print('Cleaned up and ready in: {}!'.format(upath_))

# coco-2014-test
print('Downloading: {}'.format(COCO_2014_TEST_IMAGES_URL))
dpath_ = os.path.join(output_dir, 'test2014.zip')
download_file(COCO_2014_TEST_IMAGES_URL, dpath_)
upath_ = os.path.join(output_dir, 'images', 'test2014')
unzip_file(dpath_, upath_)
print('Cleaned up and ready in: {}!'.format(upath_))

# coco-2017-train
print('Downloading: {}'.format(COCO_2017_TRAIN_IMAGES_URL))
dpath_ = os.path.join(output_dir, 'train2017.zip')
download_file(COCO_2017_TRAIN_IMAGES_URL, dpath_)
upath_ = os.path.join(output_dir, 'images', 'train2017')
unzip_file(dpath_, upath_)
print('Cleaned up and ready in: {}!'.format(upath_))

# coco-2017-val
print('Downloading: {}'.format(COCO_2017_VAL_IMAGES_URL))
dpath_ = os.path.join(output_dir, 'val2017.zip')
download_file(COCO_2017_VAL_IMAGES_URL, dpath_)
upath_ = os.path.join(output_dir, 'images', 'val2017')
unzip_file(dpath_, upath_)
print('Cleaned up and ready in: {}!'.format(upath_))

# coco-2017-test
print('Downloading: {}'.format(COCO_2017_TEST_IMAGES_URL))
dpath_ = os.path.join(output_dir, 'test2017.zip')
download_file(COCO_2017_TEST_IMAGES_URL, dpath_)
upath_ = os.path.join(output_dir, 'images', 'test2017')
unzip_file(dpath_, upath_)
print('Cleaned up and ready in: {}!'.format(upath_))
import os
import json
import webdataset as wds
from tqdm import tqdm
import argparse

##### Individual write function
def write_sample(writer, sample, key):
    sample_data = {"__key__": key}
    
    # Handle images
    for idx, image in enumerate(sample['images']):
        if image is not None:
            img_path = image.replace(REPLACE_PATH, IMAGES_PATH)

            # we assume the image exists!
            # we could try catch this if we want, but for now we will let it verbosely fail if images don't exist.
            with open(img_path, "rb") as stream:
                bindata = stream.read()
            
            extension = img_path.split(".")[-1]
            assert extension in ["jpg", "png"], "Unknown extension: {}!!!".format(extension)

            sample_data["{}.{}".format(idx, extension)] = bindata            

    # Handle texts
    for idx, ocr in enumerate(sample['text_ocr_list']):
        if ocr is not None:
            sample_data["{}.txt".format(idx)] = ocr.encode("utf-8")
    
    # Verify
    num_samps_written = len(sample_data)  # number of samples that we've written to the current dict
    num_samps_to_match = len(sample["images"]) + 1  # total number of samples of images (multiplied by 2 since we also write metajsons per image) + total number of texts + 1 (for the sample __key__) 
    assert num_samps_written == num_samps_to_match, "Wrong number of files written!!! sample_data: {} | sample: {}".format(num_samps_written, num_samps_to_match)
    samp_data_imgs = [] 
    for x in list(sample_data.keys()):
        if x.split(".")[-1] in ["jpg", "png"]:
            samp_data_imgs.append(x)
    sample_imgs = [x for x in sample["images"] if x is not None]
    assert len(samp_data_imgs) == len(sample_imgs), "Wrong number of images written!!! sample_data_images: {} | sample_images: {}".format(len(samp_data_imgs), len(sample_imgs))
    samp_data_txts = [x for x in list(sample_data.keys()) if x.endswith(".txt")]
    sample_txts = [x for x in sample["text_ocr_list"] if x is not None]
    assert len(samp_data_txts) == len(sample_txts), "Wrong number of texts written!!! sample_data_texts: {} | sample_texts: {}".format(len(samp_data_txts), len(sample_txts))

    # Write to shard
    writer.write(sample_data)

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--start_sample", type=int, default=0, help="which sample to start from")
parser.add_argument("--end_sample", type=int, default=600_000, help="which sample to finish at")
parser.add_argument("--max_samples_per_shard", type=int, default=10_000, help="how many samples per shard")
parser.add_argument("--json_file", type=str, required=True, help="multimodal_textbook.json from the original release")
parser.add_argument("--images_dir", type=str, required=True, help="folder with the uncompressed dataset images")
parser.add_argument("--output_dir", type=str, required=True, help="where to write the resharded tars")
args = parser.parse_args()

# Which chunk do we load
START_SAMPLE = args.start_sample
END_SAMPLE = args.end_sample

assert 0 <= START_SAMPLE <= END_SAMPLE, "Wrong chunking args passed!"

# First load the json file
JSON_FILE = args.json_file
with open(JSON_FILE) as f:
    data = json.load(f)

data = data[START_SAMPLE : END_SAMPLE]

# Path where images are uncompressed (this will be replaced in the image paths of the jsons directly)
IMAGES_PATH = os.path.join(args.images_dir, "")

# Path to replace while reading the json file image paths
REPLACE_PATH = "/mnt/workspace/zwq_data/interleaved_dataset/"

# Path to write the resharded tars to
OUT_PATH = os.path.join(args.output_dir, "shards_{}_{}".format(START_SAMPLE, END_SAMPLE))
os.makedirs(OUT_PATH, exist_ok=True)
sink_pattern = os.path.join(OUT_PATH, f"%04d.tar")

# Track samples and shards
sample_count = 0
shard_count = 0
MAX_SAMPLES_PER_SHARD = args.max_samples_per_shard

# Start writing
print("Processing {} samples to write to {}".format(len(data), OUT_PATH))
with wds.ShardWriter(sink_pattern, maxcount=MAX_SAMPLES_PER_SHARD, maxsize=100e9) as writer:

    # Iterate over individual samples
    for sample_index, sample in tqdm(enumerate(data), total=len(data), ascii=True):

        # create unique key for sample
        key = f"{sample_index+START_SAMPLE:04d}"

        # write to the shard
        write_sample(writer, sample, key)
        sample_count += 1

        # logging
        if sample_count % MAX_SAMPLES_PER_SHARD == 0:
            print(f"Shard {shard_count} completed with {MAX_SAMPLES_PER_SHARD} samples")
            shard_count += 1

print(f"Total samples processed: {sample_count} into {shard_count} shards!")
print(f"Full resharded dataset available at: {OUT_PATH}")
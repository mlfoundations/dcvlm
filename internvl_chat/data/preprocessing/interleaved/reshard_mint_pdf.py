'''
In this script, we are resharding the MINT-1T-PDF split into a usable format for our multimodal training.

First, we select a 10% random chunk of the dataset that we want to process since this is the scale we operate at.
Therefore, we select 53K tars out of 530K total tars that are available on HF. The overall PDF split is 33TB, hence
we roughly process 3TB and reshard.

Additionally, this script can be run on multiple processes independently. To enable that, the script takes in 
`start_tar` and `end_tar` arguments. Based on these arguments, the script will only process the subset of the
10% dataset.

So overall, 

Original dataset ---> 33TB, ~532K tars

10% subset dataset we process ---> 3TB, ~53K tars

The `start_tar` and `end_tar` arguments of this script will basically operate over the 35K tars.
'''
import os
import json
import webdataset as wds
from tqdm import tqdm
import argparse
import io
import time
from PIL import Image
import random
random.seed(0)  # for consistency

##### Utility functions
def decode_tiff(key, data):
    if key.endswith(".tiff") or key.endswith(".tif"):
        try:
            return Image.open(io.BytesIO(data))
        except Exception as e:
            print(f"Failed to decode TIFF image ==> {key}: {e}")
            return None
    return None

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def write_sample(writer, sample, key):

    # Write to the dict we want to return
    sample_data = {"__key__": key}
    
    # Extract the tiff image & the image and text multimodal sequences
    text_sequence = sample["json"]["texts"]  # this should be of form [null, txt1, txt2, null, ...], where nulls represent images and txts are texts
    image_sequence = sample["json"]["images"]  # this should be of form [img1, null, null, img2, ...], where nulls represent texts and imgs are images
    assert len(text_sequence) == len(image_sequence), "Image and text multimodal sequences are of unequal lengths!"
    tiff_image = sample["tiff"]

    # Write images to dict
    all_img_bytes = []
    frame_number = 0
    while True:
        buffer = io.BytesIO()  # Create an in-memory bytes buffer
        tiff_image.save(buffer, format="PNG")  # Save the current frame to the buffer as PNG
        bindata = buffer.getvalue()  # Get the bytes data
        all_img_bytes.append(bindata)  # Append to img list
        frame_number += 1  # Update current frame                
        try:
            tiff_image.seek(frame_number)  # Move to the next frame
        except EOFError:
            break

    assert len(all_img_bytes) == len([x for x in image_sequence if x is not None]), "Images written from tiff and image sequence don't match!!!"

    pt_ = 0
    for idx, image in enumerate(image_sequence):
        if image is not None:
            sample_data["{}.{}".format(idx, "png")] = all_img_bytes[pt_]
            pt_ += 1

    # Write texts to dict
    for idx, text in enumerate(text_sequence):
        if text is not None:
            sample_data["{}.txt".format(idx)] = text.encode("utf-8")

    # Verify
    num_samps_written = len(sample_data)  # number of samples that we've written to the current dict
    num_samps_to_match = len(image_sequence) + 1  # total number of samples of images (multiplied by 2 since we also write metajsons per image) + total number of texts + 1 (for the sample __key__) 
    assert num_samps_written == num_samps_to_match, "Wrong number of files written!!! sample_data: {} | sample: {}".format(num_samps_written, num_samps_to_match)
    samp_data_imgs = [] 
    for x in list(sample_data.keys()):
        if x.split(".")[-1] in ["jpg", "png"]:
            samp_data_imgs.append(x)
    sample_imgs = [x for x in image_sequence if x is not None]
    assert len(samp_data_imgs) == len(sample_imgs), "Wrong number of images written!!! sample_data_images: {} | sample_images: {}".format(len(samp_data_imgs), len(sample_imgs))
    samp_data_txts = [x for x in list(sample_data.keys()) if x.endswith(".txt")]
    sample_txts = [x for x in text_sequence if x is not None]
    assert len(samp_data_txts) == len(sample_txts), "Wrong number of texts written!!! sample_data_texts: {} | sample_texts: {}".format(len(samp_data_txts), len(sample_txts))

    # Write to shard
    writer.write(sample_data)

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--start_tar", type=int, default=0, help="which tar to start from")
parser.add_argument("--end_tar", type=int, default=1, help="which tar to finish at")
parser.add_argument("--max_samples_per_shard", type=int, default=10_000, help="how many samples per shard")
parser.add_argument("--subset_percent", type=float, default=0.1, help="how much of the full dataset to reshard, default is 10%")
parser.add_argument("--sample_log_frequency", type=int, default=100, help="after how many samples to log status to console")
parser.add_argument("--input_dir", type=str, required=True, help="folder with the original MINT-1T-PDF-CC-* super-shards")
parser.add_argument("--output_dir", type=str, required=True, help="where to write the resharded tars")
args = parser.parse_args()

# Which chunk do we load
START_SHARD = args.start_tar
END_SHARD = args.end_tar
assert 0 <= START_SHARD <= END_SHARD, "Wrong chunking args passed!"

# First load all available tars
ORIGINAL_SHARDS_FOLDER = args.input_dir
original_super_shards = [os.path.join(ORIGINAL_SHARDS_FOLDER, x) for x in os.listdir(ORIGINAL_SHARDS_FOLDER) if "MINT-1T-PDF-CC-" in x]
original_shards = []
for super_shard in original_super_shards:
    original_shards.extend([os.path.join(super_shard, x) for x in os.listdir(super_shard) if "-shard-" in x])
print("There are {} total shards available!".format(len(original_shards)))

# Shuffle the shards 
random.shuffle(original_shards)
shards_to_process = original_shards
print("Total shards that we have to process overall: {}".format(len(shards_to_process)))

# Now get the individual tars and subset them to `subset_percent` we have to process
original_tars = []
for shardfolder in shards_to_process:
    original_tars.extend([os.path.join(shardfolder, x) for x in os.listdir(shardfolder) if ".tar" in x])
print("Total tars that we have available: {}".format(len(original_tars)))
tars_to_process = original_tars[ : int(args.subset_percent * len(original_tars)) ]
print("Total tars that we have to process: {}".format(len(tars_to_process)))

# Get current tars that we have to process given the start_tar and end_tar params
current_tars_to_process = tars_to_process[ START_SHARD : END_SHARD ]
print("Total tars that current script has to process: {}".format(len(current_tars_to_process)))

# Path to write the resharded tars to
OUT_PATH = os.path.join(args.output_dir, "shards_{}_{}".format(START_SHARD, END_SHARD))
os.makedirs(OUT_PATH, exist_ok=True)
sink_pattern = os.path.join(OUT_PATH, f"%06d.tar")

# Track samples and shards
sample_count = 0
shard_count = 0
MAX_SAMPLES_PER_SHARD = args.max_samples_per_shard

# track time for logging
start_time = time.time()

# Start writing
print("Processing {} tars to write to {}".format(len(current_tars_to_process), OUT_PATH))
with wds.ShardWriter(sink_pattern, maxcount=MAX_SAMPLES_PER_SHARD, maxsize=100e9) as writer:

    # Get string to pass to webdataset iterator
    str_ = "::".join(current_tars_to_process)

    # Create webdataset and iterator
    ds = wds.WebDataset(str_, shardshuffle=False).decode(decode_tiff, handler=log_and_continue)
    ds_iter = iter(ds)

    # Iterate over webdataset
    while True:
        try:
            sample = next(ds_iter)
            key = sample['__key__']

            # write to the shard
            write_sample(writer, sample, key)
            sample_count += 1

            # logging
            if sample_count % args.sample_log_frequency == 0:
                print("Written {} samples in {} seconds".format(sample_count, time.time()-start_time))

            if sample_count % MAX_SAMPLES_PER_SHARD == 0:
                print(f"Shard {shard_count} completed with {MAX_SAMPLES_PER_SHARD} samples")
                shard_count += 1

        except StopIteration:
            break

print(f"Total samples processed: {sample_count} into {shard_count} shards!")
print(f"Full resharded dataset available at: {OUT_PATH}")
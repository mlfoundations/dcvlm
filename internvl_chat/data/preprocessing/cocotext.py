###################################
# 1. download the cocotext dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

import pandas as pd
import os
import io
from PIL import Image
import random
import wget

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import zipfile
from tqdm import tqdm


"""
These prompts were generated using GPT-4O using the following prompt:
I want to now create a set of prompts from which I can sample for prompting the model to perform the task of text detection and recognition. For example, a simple valid prompt would be: "Please find all text present in the image." Another valid prompt would be: "Perform the task of text detection and recognition and output all text and bounding boxes that you see in the image, in sequence."
Could you please give a python list of 10-15 prompts which I can use as training prompts for constructing my OCR dataset.
"""
OCR_PROMPTS = [
    "Please find all text present in the image.",
    "Perform the task of text detection and recognition and output all text and bounding boxes that you see in the image, in sequence.",
    "Identify and extract all textual content from the given image, including location details.",
    "Detect all text in the image and return the recognized characters along with their bounding boxes.",
    "Extract every word and character present in the image along with their precise positions.",
    "Locate all instances of text in the image and transcribe them accurately.",
    "Analyze the image and retrieve all text along with its spatial coordinates.",
    "Detect and recognize any visible text in the provided image, specifying its location.",
    "Extract the text appearing in the image and provide the bounding box for each instance.",
    "Perform text detection and recognition on the image, ensuring accurate transcription and spatial mapping.",
    "Identify all words in the image and return their bounding boxes in a structured format.",
    "Find and extract all text elements in the image, maintaining their reading order.",
    "Recognize and transcribe any text appearing in the image, including its exact placement.",
    "Locate all occurrences of text within the image and provide structured output with bounding boxes.",
    "Perform OCR on the given image and return all detected text along with their corresponding positions.",
]
"""
In the same chat:
Now generate some perturbations of the possible ai response:
Here is the location of each of the transcribed texts from the image
"""
AI_RESPONSES = [
    "Here are the locations of the transcribed texts from the image:",
    "Below are the bounding boxes for each detected text instance in the image:",
    "These are the coordinates for the recognized text in the image:",
    "The following are the positions of all transcribed text segments:",
    "Here is a list of detected text along with their respective locations:",
    "Here are the extracted text entries along with their spatial coordinates:",
    "Displaying the locations of all recognized text elements in the image:",
    "The detected text instances and their corresponding positions are listed below:",
    "Here is where each transcribed text appears within the image:",
    "Providing the exact positions for the recognized text in the image:",
    "Below, you will find the spatial details of each detected text element:",
    "All transcribed text locations are given in the following list:",
    "Here are the identified text regions and their bounding coordinates:",
    "Here is the placement of each recognized text within the image:",
    "Listing the bounding box coordinates for all detected text elements:",
]


# path where files are temporarily stored
DOWNLOAD_DIR = "cocotext_files"
# output dir
OUTPUT_DIR = "./cocotext"


def normalize_coordinates(box, image_width, image_height):
    """
    Code copied from https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data
    """
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000),
    ]
    return normalized_box


def main():
    # check if COCO_DIR environment variable is set
    assert (
        "COCO_DIR" in os.environ
    ), "Please set-up your COCO_DIR env variable for the path to all coco image folders, set it using export COCO_DIR=XXXX'"

    # download annotations with wget from original source
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    downloaded_path = os.path.join(DOWNLOAD_DIR, "cocotext.v2.zip")
    wget.download(
        "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip",
        out=downloaded_path,
    )
    print()

    # extract annotations into download dir
    with zipfile.ZipFile(downloaded_path, "r") as fp:
        fp.extractall(DOWNLOAD_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(DOWNLOAD_DIR, "cocotext.v2.json")) as fp:
        annotations = json.load(fp)

    # annotations are in this format
    # annotation_id -> img_id, text, box, legible
    #
    # I want something like
    # img_id -> annotation_id, text, box, legible
    # This allows to create a single conversation out of every image
    rev_annotations = {}

    # I used these variable to check dataset statistics. You can remove the following commented lines.
    # set_images = set()
    # train_found_count = 0
    for ann_id in annotations["anns"].keys():
        # get image id for each annotation
        image_id = annotations["anns"][ann_id]["image_id"]

        # if new image id create list in the new annotation dict
        if image_id not in rev_annotations:
            rev_annotations[image_id] = []

        # append annotation to image_id list
        # note some files (e.g., image_id) are redundant.
        # As it is just a temporary dict, I do not remove them to avoid unnecessary bugs
        rev_annotations[image_id].append(annotations["anns"][ann_id])

        #######################################################################################################################
        # These lines below were used to compute dataset statistcs
        # First I populate a set with image ids (so to have unique image filenames)
        # I check that the image was present in either of the two directories (train, val)
        # Then I count how many of these are found in the training split (spoiler all of them)
        #
        #   set_images.add(image_id)
        #   train_found = os.path.exists(f"/data/tdemin/datasets/coco/train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg")
        #   val_found = os.path.exists(f"/data/tdemin/datasets/coco/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg")
        #   if train_found:
        #       train_found_count += 1
        #
        # here I make sure that the file name is found in only one of the two directories
        # it's not necessary but I wanted to be sure
        #   assert train_found != val_found
        #
        # Result
        # Only train images found
        # 23485 unique images vs 63686 reported by authors
        # 201126 annotations vs 239506 reported by authors
        #######################################################################################################################

    # now that the dict with image ids as keys is created, I can generate conversations
    new_jsonl_entries = []
    cnt = -1
    illegible = 0
    # iterate over all images in the annotations
    for image_id in tqdm(rev_annotations):
        # COCO-Text2.0 uses COCO2014
        # coco 2014 original filenames are in this format "COCO_{train/val}2014_012345678910.jpg"
        # since all images are from the train split I hard coded the relative path
        # CHECK IF THIS IS THE CORRECT FORMAT FOR THIS PROJECT (it should work on a vanilla download of COCO214)
        # NOTE: (Matteo) chaned this line according to how things are organized on disk in our case, but the script looks good
        image_path = os.path.join("images", "train2014", "train2014", f"COCO_train2014_{str(image_id).zfill(12)}.jpg")

        # check if path exists
        assert os.path.join(
            os.environ["COCO_DIR"], image_path
        ), "Image {} does not exist!!!".format(image_path)

        # open image to get sizes
        image = Image.open(os.path.join(os.environ["COCO_DIR"], image_path))
        width, height = image.size

        # build user prompt by randomly chosing <image> position and user prompt
        image_beginning = random.choice([0, 1])
        prompt = random.choice(OCR_PROMPTS)
        if image_beginning:
            conversations = [{"from": "human", "value": f"<image>{prompt}"}]
        else:
            conversations = [{"from": "human", "value": f"{prompt}<image>"}]

        # build gpt output by randomly chosing the gpt answer
        response = random.choice(AI_RESPONSES)
        at_least_one_legible = False # discard images that only contain illegible text
        
        # let's iterate over each annotation of the image
        for ann in rev_annotations[image_id]:
            # proceed only if legible
            if ann["legibility"] == "legible":
                at_least_one_legible = True

                # normalize boxes between 0 and 1000
                box = ann["bbox"]
                normalized_box = normalize_coordinates(box, width, height)

                # append reference text and bbox to response
                response += f"\n<ref>{ann['utf8_string']}</ref><box>[{normalized_box}]</box>"

        # check if there is at least one legible text in the image, if not skip image
        if not at_least_one_legible:
            illegible += 1
            continue
        
        # add gpt answer to conversation
        cnt += 1
        conversations.append({"from": "gpt", "value": response})

        entry_ = {
            "id": cnt,
            "image": image_path,
            "height": height,
            "width": width,
            "conversations": conversations,
        }
        new_jsonl_entries.append(entry_)

    output_file = os.path.join(OUTPUT_DIR, "preprocessed.jsonl")
    with open(output_file, "w") as fp:
        for entry in new_jsonl_entries:
            fp.write(json.dumps(entry) + "\n")

    print(
        f"Preprocessing complete! Saved as {output_file} with 0 saved images and {cnt+1} total samples. Discarded {illegible} illegibles images."
    )

    shutil.rmtree(DOWNLOAD_DIR)


if __name__ == "__main__":
    main()

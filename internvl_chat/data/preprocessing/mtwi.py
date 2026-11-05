###################################
# 1. download the mtwi dataset using the snapshot_download below [note: it may be beneficial to use hf_transfer if you like, see here: https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads]
# 2. run this script from the same folder to get the preprocessed version!
###################################

from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
import gdown
import numpy as np

random.seed(0)  # to ensure reproducibility while creating the dataset jsonl
import json
import shutil
import zipfile


# path where files are temporarily stored
DOWNLOAD_DIR = "./mtwi_files"
# output dir
OUTPUT_DIR = "./mtwi"
# image output dir
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


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


def normalize_coordinates(box, image_width, image_height):
    """
    Code copied from https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000),
        round((x3 / image_width) * 1000),
        round((y3 / image_height) * 1000),
        round((x4 / image_width) * 1000),
        round((y4 / image_height) * 1000),
    ]
    return normalized_box


def is_rectangle_and_rotated(corners):
    """
    Checks if the given bounding box is a rectangle and whether it is rotated.
    :param corners: A list of 8 elements representing the x and y coordinates of the four corners
                    in the order: (up left x, up left y, down left x, down left y,
                                   down right x, down right y, up right x, up right y)
    :return: (is_rectangle, is_rotated)
    """
    if len(corners) != 8:
        raise ValueError("Exactly eight elements are required.")

    # Convert to NumPy array for vectorized computations
    points = np.array(corners).reshape(4, 2)

    # Compute pairwise distances (side lengths)
    dists = [np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)]

    # Check if opposite sides are equal (parallelogram condition)
    if not np.isclose(dists[0], dists[2]) or not np.isclose(dists[1], dists[3]):
        return False, False

    # Compute diagonals
    diag1 = np.linalg.norm(points[0] - points[2])
    diag2 = np.linalg.norm(points[1] - points[3])

    # If diagonals are equal, it's a rectangle
    is_rectangle = np.isclose(diag1, diag2)

    if not is_rectangle:
        return False, False

    # Check if the edges are axis-aligned
    vec1 = points[1] - points[0]  # Vector from up-left to down-left
    vec2 = points[3] - points[0]  # Vector from up-left to up-right

    # If either vector has a nonzero component in both x and y, it's rotated
    is_rotated = not (np.isclose(vec1[0], 0) or np.isclose(vec1[1], 0))

    return True, is_rotated


def corners_to_rectangle(corners):
    """
    Top left corner (x, y) is obtained as min x and min y
    Bottom right corner (x+w, y+h) is obtained as the max x and max y
        ______
     __/      |
    |         |
    |_________|

    =>
     _________
    |         |
    |         |
    |_________|
    """
    x = min(corners[::2])
    y = min(corners[1::2])
    w = max(corners[::2]) - x
    h = max(corners[1::2]) - y
    return x, y, w, h


def quadrilateral_area(points):
    """Compute the area of a quadrilateral using the shoelace formula."""
    x = points[::2]
    y = points[1::2]

    # Shoelace formula for a 4-sided polygon
    area = 0.5 * abs(
        x[0] * y[1]
        + x[1] * y[2]
        + x[2] * y[3]
        + x[3] * y[0]
        - (y[0] * x[1] + y[1] * x[2] + y[2] * x[3] + y[3] * x[0])
    )
    return area


def rectangle_area(points):
    # x, y, w, h -> w * h
    return points[2] * points[3]


def compute_iou(corners_area, box_area):
    assert box_area >= corners_area
    # box encloses corners
    # thus corners area is the intersection
    # and box are is the union
    intersection = corners_area
    union = box_area
    return intersection / union


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    part1_download_path = os.path.join(DOWNLOAD_DIR, "part1.zip")
    part2_download_path = os.path.join(DOWNLOAD_DIR, "part2.zip")
    gdown.download(
        url="https://drive.google.com/file/d/1DDVa81xldAAy9iSAggvNvAIXhCVyBnQo/view?usp=drive_link",
        output=part1_download_path,
        fuzzy=True
    )
    gdown.download(
        url="https://drive.google.com/file/d/1MNZ3lVZzj6oXBkMKLRb_qdBenWF1s5ew/view?usp=drive_link",
        output=part2_download_path,
        fuzzy=True
    )

    with zipfile.ZipFile(part1_download_path, "r") as fp:
        fp.extractall(DOWNLOAD_DIR)
    with zipfile.ZipFile(part2_download_path, "r") as fp:
        fp.extractall(DOWNLOAD_DIR)

    # part 1 extracts everything into DOWNLOAD_DIR/train_1000
    # part 2 extracts everything into DOWNLOAD_DIR
    part_1_imgs = os.path.join(DOWNLOAD_DIR, "train_1000/image_1000")
    part_1_txt = os.path.join(DOWNLOAD_DIR, "train_1000/txt_1000")
    part_2_imgs = os.path.join(DOWNLOAD_DIR, "image_9000")
    part_2_txt = os.path.join(DOWNLOAD_DIR, "txt_9000")

    unique_image_names = set()
    new_jsonl_entries = []
    cnt = -1  # conversation number
    imgs_saved = 0  # number of images saved
    illegible = 0  # number of images that do not contain legible text
    illegible_ann = 0  # number of illegible annotations
    rotated = 0  # number of rotated rectangles
    not_rectangles = 0  # number of annotations that are not rectangle
    not_rotated_and_rectangles = 0  # number of annotations that are legit
    all_annotations = 0  # total number of annotations
    min_area = 1000**2 # max area is entire normalized image
    mean_area = []

    # There are 10000k annotations in the dataset.
    # I hardcoded it to have a single progress bar
    bar = tqdm(range(10000), ascii=True)

    # iterate over the two parts
    for part_num, (img_dir, txt_dir) in enumerate(
        ((part_1_imgs, part_1_txt), (part_2_imgs, part_2_txt))
    ):
        # iterate image directory to retrieve filenames
        for img_num, img_name in enumerate(sorted(os.listdir(img_dir))):
            bar.update(1)

            # remove extension to obtain filename
            # cannot do replace nor split as there are annotations like dshjkdshdshj.jpg.txt or T1..dhdshjd.jpg
            filename = img_name[:-4]

            image_path = os.path.join(img_dir, img_name)
            image = Image.open(image_path)
            width, height = image.size

            # image unique name
            image_output_name = os.path.join(
                OUTPUT_IMAGE_DIR, f"part_{part_num}_sample_{img_num}.png"
            )
            assert (
                image_output_name not in unique_image_names
            ), f"Duplicate image path {image_output_name}."
            unique_image_names.add(image_output_name)

            txt_path = os.path.join(txt_dir, f"{filename}.txt")
            with open(txt_path, "r") as fp:
                annotations = fp.readlines()
            # remove newline
            annotations = [a[:-1] for a in annotations]

            # build user prompt by randomly chosing <image> position and user prompt
            image_beginning = random.choice([0, 1])
            prompt = random.choice(OCR_PROMPTS)
            if image_beginning:
                conversations = [{"from": "human", "value": f"<image>{prompt}"}]
            else:
                conversations = [{"from": "human", "value": f"{prompt}<image>"}]

            # build gpt output by randomly chosing the gpt answer
            response = random.choice(AI_RESPONSES)
            at_least_one_legible = False  # discard images that only contain illegible text

            corners_list = []
            ious = []
            corner_areas = []
            box_areas = []
            # let's iterate over each annotation of the image
            for ann in annotations:
                all_annotations += 1

                # csv -> list
                ann = ann.split(",")

                # first 8 elements are corners, convert first ones into float
                corners = [float(corner) for corner in ann[:8]]
                text = " ".join(ann[8:])  # concat text annotation

                # if text == "###" it means it is illegible
                if text == "###":
                    illegible_ann += 1
                    continue
                at_least_one_legible = True

                # normalize the coordinate between 0 and 1000 and convert to int (should be more robust to subtle corner changes)
                corners = normalize_coordinates(corners, width, height)

                # check if is a rectangle and if is rotated
                is_rectangle, is_rotated = is_rectangle_and_rotated(corners)

                if not is_rectangle:
                    not_rectangles += 1
                elif is_rotated:
                    rotated += 1
                else:
                    not_rotated_and_rectangles += 1

                box = corners_to_rectangle(corners)
                corners_area, box_area = quadrilateral_area(corners), rectangle_area(box)
                iou = compute_iou(corners_area, box_area)
                if corners_area < min_area:
                    min_area = corners_area
                mean_area.append(corners_area)

                # append reference text and bbox to response
                box_processed = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                response += f"\n<ref>{text}</ref><box>[{box_processed}]</box>"
                corners_list.append(corners)
                ious.append(iou)
                corner_areas.append(corners_area)
                box_areas.append(box_area)

            # check if there is at least one legible text in the image, if not skip image
            if not at_least_one_legible:
                illegible += 1
                continue

            # add gpt answer to conversation
            cnt += 1
            conversations.append({"from": "gpt", "value": response})

            entry_ = {
                "id": cnt,
                "image": os.path.join("images", os.path.basename(image_output_name)),
                "height": height,
                "width": width,
                "conversations": conversations,
                # "corners": corners_list,
                # "ious": ious,
                # "corner_areas": corner_areas,
                # "box_areas": box_areas
            }
            new_jsonl_entries.append(entry_)

            # save image only if at least one legible
            image.save(image_output_name)
            imgs_saved += 1

    output_file = os.path.join(OUTPUT_DIR, "preprocessed.jsonl")
    with open(output_file, "w") as fp:
        for entry in new_jsonl_entries:
            fp.write(json.dumps(entry) + "\n")

    print(
        f"Preprocessing complete! Saved as {output_file} with {imgs_saved} saved images in {OUTPUT_IMAGE_DIR} and {cnt+1} total samples with {not_rotated_and_rectangles} legible and correct bboxes. Discarded {illegible} illegible images, {illegible_ann} illegible annotations. Converted {not_rectangles} bboxes that are not rectangles, and {rotated} rotated bboxes. Total annotations in the dataset {all_annotations}. Min area {min_area}. Mean area {sum(mean_area) / len(mean_area):.2f}"
    )

    shutil.rmtree(DOWNLOAD_DIR)


if __name__ == "__main__":
    main()

import wget
import os
import json
import random
import zipfile


DATASET_NAME = "textocr"
os.makedirs(os.path.join(DATASET_NAME, "images"), exist_ok=True)

IMAGE_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
IMAGE_ZIP = "train_val_images.zip"
ANN_URL = "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json"
ANN_FILE = "TextOCR_0.1_train.json"

print(f"Downloading images to '{os.path.join(DATASET_NAME, IMAGE_ZIP)}'")
wget.download(IMAGE_URL)
print(f"Extracting images to '{os.path.join(DATASET_NAME, 'images')}'")
with zipfile.ZipFile(IMAGE_ZIP) as zip:
    zip.extractall(os.path.join(DATASET_NAME, "images"))
print("Downloading annotations")
if not os.path.exists(ANN_FILE):
    wget.download(ANN_URL, out=ANN_FILE)


# Preprocessing

# The initial prompt and response are derived from the example in
# https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data
# The "all objects" is replaced with "all text", and all subsequent variations are retrieved by asking ChatGPT for paraphrases
# During preprocessing, I omit the response preamble sometimes
OCR_PROMPTS_AND_RESPONSES = [
    [
        "Please detect and label all text in the following image and mark their positions.",
        "Sure, I will detect and label all text in the image and mark their positions."
    ],
    [
        "Identify and tag all text in the given image, specifying their locations.",
        "Sure, I will identify and tag all text in the image while specifying their locations."
    ],
    [
        "Find and annotate all text in this image, noting their positions.",
        "Certainly, I will find and annotate all text in the image and note their positions."
    ],
    [
        "Locate and label all text present in the image, marking where they appear.",
        "Of course, I will locate and label all text in the image, marking where they appear."
    ],
    [
        "Detect and highlight every text element in the image, including bounding boxes.",
        "Got it! I will detect and highlight every text element in the image including bounding boxes."
    ],
    [
        "Recognize and label all textual content in the image, specifying their coordinates.",
        "Understood! I will recognize and label all textual content in the image and specify their coordinates."
    ],
    [
        "Extract and tag all text from the image, marking their exact locations.",
        "Here is all the text from the image, with their exact locations:"
    ],
    [
        "Scan the image for text and label each instance, including its position.",
        "Absolutely! I will scan the image for text and label each instance along with its position."
    ]
]

# Function to convert bbox format from xywh to xyxy
# COCO by default uses x_min, y_min, w, h format
# InternVL 2.5 format requires x_min, y_min, x_max, y_max format
def convert_bbox_xywh_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

# Function to normalize bbox coordinates, taken from InternVL documentations
def normalize_coordinates(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    normalized_bbox = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_bbox


print("Starting preprocessing")
random.seed(0)

out_data = []
with open(ANN_FILE) as fp:
    all_data = json.load(fp)

for image_id, image_data in all_data['imgs'].items():
    if image_data['set'] != "train":
        continue
    text_annotations = []
    for ann_id in all_data['imgToAnns'][image_id]:
        ann_data = all_data['anns'][ann_id]
        text_annotations.append("<ref>{}</ref><box>[{}]</box>".format(
            ann_data['utf8_string'],
            normalize_coordinates(
                convert_bbox_xywh_xyxy(ann_data['bbox']),
                image_data['width'], image_data['height']
            )
        ))
    human_prompt, gpt_prefix = random.choice(OCR_PROMPTS_AND_RESPONSES)
    gpt_prefix = gpt_prefix + "\n\n" if random.randrange(2) == 0 else ""
    image_conversations = [
        {'from': "human", 'value': human_prompt},
        {'from': "gpt", 'value': gpt_prefix + "\n".join(text_annotations) },
    ]
    if image_conversations:
        if random.randrange(2) == 0:
            image_conversations[0]['value'] = "<image>\n" + image_conversations[0]['value']
        else:
            image_conversations[0]['value'] = image_conversations[0]['value'] + "\n<image>"
        out_data.append({
            'id': image_data['id'],
            'image': os.path.join("images/train_images", os.path.basename(image_data['file_name'])),
            'width': image_data['width'],
            'height': image_data['height'],
            'conversations': image_conversations
        })

with open(os.path.join(DATASET_NAME, "preprocessed.jsonl"), "w") as fp:
    for example in out_data:
        print(json.dumps(example), file=fp)
imgs_saved = len(os.listdir(os.path.join(DATASET_NAME, "images/train_images")))

print("Preprocessing complete! Saved as '{0}/preprocessed.jsonl' with {1} saved images in '{0}/images' and {2} total samples.".format(DATASET_NAME, imgs_saved, len(out_data)))

os.remove("train_val_images.zip")
os.remove("TextOCR_0.1_train.json")

import random
import logging
from PIL import Image
from internvl.train.constants import (
    EOL_SEPARATOR, 
    LLAVA_DESC_SHORT_PROMPTS,
    DATASET_FORMAT_ROUTERS
)


def txt2conversations(txt: str) -> list[dict]:
    conv_lines = txt.split(EOL_SEPARATOR)
    conversations = []
    for line in conv_lines:
        # by construction, the 'role' is prepended
        # to the actual value, followed by a column
        role = line.split(":")[0]

        # by construction, the rest of the line is the value
        # (the +1 is to remove the manually added column)
        value = line[(len(role)+1) :]

        # we can then use role and value to create the dict for this turn
        turn = {"from": role, "value": value}
        conversations.append(turn)
    return conversations    


def format_instruction(sample: dict) -> dict:
    conversations_txt = sample.pop("conversations.txt")
    sample["conversations"] = txt2conversations(conversations_txt)
    return sample


def format_mmdoc_ic(sample: dict) -> dict:
    multimodal_sequence = [v for k, v in sample.items() if k.split(".")[0].isdigit()]

    # If the last item in the sequence is an image, we drop it
    # (we cannot have loss over it).
    while isinstance(multimodal_sequence[-1], Image.Image):
        multimodal_sequence = multimodal_sequence[:-1]

    assert not isinstance(multimodal_sequence[-1], Image.Image), "Last item is an image"
    assert len(multimodal_sequence) > 1, "Single-item multimodal sequence after popping images, invalid and dropping this...!"

    # after dropping last-item images, make sure 
    # we still have at least one image in the document
    if not any(isinstance(item, Image.Image) for item in multimodal_sequence):
        raise ValueError(
            "No images left in this multimodal sequence, skipping."
        )

    # Construct the conversation in ChatML format.
    # Everything until the last text part of the sequence is in "human".
    # The last text chunk is "gpt" (over which we take the loss)
    human_part = ""
    gpt_part = ""
    delimiters = ["\n", " ", "\t", "."]
    for mm_index, v in enumerate(multimodal_sequence):
        # last chunk is "gpt"
        if mm_index == len(multimodal_sequence) - 1:
            gpt_part = v
        # previous chunks are "human"
        else:
            delimiter = random.choice(delimiters)
            if isinstance(v, Image.Image):
                human_part += f"<image>{delimiter}" if random.choice([0, 1]) else "<image>"
            else:
                human_part += f"{v}{delimiter}" if random.choice([0, 1]) else v

    conversations = [
        {"from": "human", "value": human_part}, 
        {"from": "gpt", "value": gpt_part}
    ]
    sample["conversations"] = conversations
    return sample

def format_mmdoc_mt(sample: dict) -> dict:
    """
    Multi-Turn conversation formatting for Multimodal Documents. 
    All images fall within a human prompt, and the loss is computed over all
    text tokens of the documents. For example:
    >>> # assume an mmdoc as 0.jpg, 1.jpg, 2.txt, 3.txt, 4.txt, 5.jpg, 6.txt
    >>> conversations = [
    >>>    {"human": "<image><image>", "gpt":"2.txt 3.txt 4.txt"},
    >>>    {"human": "<image>", "gpt": "6.txt"}
    >>> ]
    """
    multimodal_sequence = [v for k, v in sample.items() if k.split(".")[0].isdigit()]

    if not any(isinstance(item, str) for item in multimodal_sequence):
        raise ValueError(
            "No text-snippets in this multimodal sequence, skipping."
        )
    
    # If the last item in the sequence is an image, we drop it
    # (we cannot have loss over it).
    while isinstance(multimodal_sequence[-1], Image.Image):
        multimodal_sequence = multimodal_sequence[:-1]

    # NOTE: perhaps we wanna keep single-item, text-only
    # sequences in the future?
    if len(multimodal_sequence) <= 1: 
        raise ValueError(
            "Single-item or empty multimodal sequence after popping images, invalid and dropping this sample."
        )
    
    if isinstance(multimodal_sequence[-1], Image.Image):
        raise ValueError(f"Last item of a multimodal document is an image, something went wrong...")
    
    # after dropping last-item images, make sure 
    # we still have at least one image in the document
    if not any(isinstance(item, Image.Image) for item in multimodal_sequence):
        raise ValueError(
            "No images left in this multimodal sequence, skipping."
        )

    # Start constructing turns of the conversation in ChatML format.
    # Every contiguous block of images is "human". 
    # Every contiguous block of texts is "gpt".
    human_part = ""
    gpt_part = ""
    delimiters = ["\n", " ", "\t", "."]
    conversations = []
    prev_is_text = False
    for item in multimodal_sequence:
        delimiter = random.choice(delimiters)
        if isinstance(item, str): # text chunk from the model
            gpt_part += f"{item}{delimiter}"
            prev_is_text = True
        elif isinstance(item, Image.Image): # image from the user
            # this is where we should close the previous turn and start a new one
            # NOTE: this accepts cases where the documents start with text
            # (i.e., the first user prompt is empty.)
            if prev_is_text:
                conversations.extend([
                    {"from": "human", "value": human_part},
                    {"from": "gpt", "value": gpt_part}
                ])
                # reset accums
                human_part = ""
                gpt_part = ""
            
            # and start the new turn
            human_part += f"<image>{delimiter}" if random.choice([0, 1]) else "<image>"
            prev_is_text = False

    # add the last turn to the conversation
    if human_part and gpt_part:
        conversations.extend([
            {"from": "human", "value": human_part},
            {"from": "gpt", "value": gpt_part}
        ])
        
    sample["conversations"] = conversations
    return sample


def format_icpairs(sample: dict) -> dict:
    '''
    Formatter for image-text pairs from img and txt keys.
    Here, we assume sample contains 'txt' and 'jpg'/'png'/'jpeg'/'webp' keys for text caption and images.
    
    Input: sample dict containing keys with texts and images
    Returns: conversation dictionary 
    '''
    # randomness
    image_beginning = random.choice([0, 1])
    random_prompt = random.choice(LLAVA_DESC_SHORT_PROMPTS)
    # handle '.txt' key in the InternVL-SA1B-Caption dataset
    gpt_part = sample['txt'] if 'txt' in sample else sample['.txt']
    # format the conversation and return
    if image_beginning:
        conversations = [
            {"from": "human", "value": f"<image>\n{random_prompt}"},
            {"from": "gpt", "value": gpt_part}
        ]
    else:
        conversations = [
            {"from": "human", "value": f"{random_prompt}\n<image>"},
            {"from": "gpt", "value": gpt_part}
        ]

    sample["conversations"] = conversations
    return sample


def conversation_formatter(sample: dict) -> dict:
    '''
    This function will convert the raw sample dict yielded by webdataset
    to a format like {__key__, conversations}

    Input: dictionary from webdataset

    Output: dictionary with conversations
    '''

    ds_name = sample["__ds_name__"]
    data_type = sample["__data_type__"]

    # formatting for instruction-tuning datasets
    # these typically come from converted jsonl datasets
    # NOTE: this is technically uneeded, we might
    # as well use the presence/absence of the 'conversations.txt' key
    # if data_type == "instruction":
    if "conversations.txt" in sample:
        return format_instruction(sample)

    if data_type == "captioning":
        return format_icpairs(sample)

    if data_type == "mmdoc_ic":
        return format_mmdoc_ic(sample)
    
    if data_type == "mmdoc_mt":
        return format_mmdoc_mt(sample)

    # in the worst case, look for an explicit formatting fn
    # for this dataset; default to captioning
    fn_name = DATASET_FORMAT_ROUTERS.get(ds_name, "format_icpairs")
    formatter = eval(fn_name)
    return formatter(sample)
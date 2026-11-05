import logging
import json
import time
import os

from copy import deepcopy
from dataclasses import dataclass, field
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from argparse import Namespace
from transformers import AutoTokenizer
from PIL import Image, ImageFile, UnidentifiedImageError, PngImagePlugin

from utils.dataloading import get_item

logger = logging.getLogger(__name__)

# some flags to avoid errors
Image.MAX_IMAGE_PIXELS = 30_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class LengthFilterDataset(Dataset):
    def __init__(self, args: Namespace):
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.output_file = make_output_path(args)
        with open(args.input_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.filter_name_char = "length_chars"
        self.filter_name_tokens = "length_tokens"
        self.filter_name_tokens_text_only = "length_tokens_text_only"
        self.filter_name_tokens_trainable_only = "length_tokens_trainable_only"
        self.filter_name_images = "length_images"
        self.filter_names = [self.filter_name_char, self.filter_name_tokens, self.filter_name_tokens_text_only, self.filter_name_tokens_trainable_only, self.filter_name_images]

        self.num_image_token = int((args.image_size // args.patch_size) ** 2 * (args.downsample_ratio ** 2))
        assert self.num_image_token == 256, "By default, the number of tokens per image-tile should be 256, please check again!"
        self.template_name = args.conv_style

        self.img_root = args.data_root
        self.ds_name = args.ds_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        img = line.get("image", None)
        if img is None:
            length_images = 0
        elif isinstance(img, str):
            length_images = 1
        elif isinstance(img, list):
            length_images = len(img) 

        conv = line["conversations"]
        length_char = get_conversation_length(conv)

        ret = get_item(
            line,
            self.template_name,
            self.img_root,
            self.tokenizer,
            None,
            ds_name=self.ds_name,
            num_image_token=self.num_image_token,
            dynamic_image_size=True,  # always use dynamic preprocessing
            use_thumbnail=True,  # always use thumbnail if we do multiple image tiles
        )
        input_ids = ret["input_ids"]
        length_token = len(input_ids)
        text_only_input_ids = ret["text_only_input_ids"]
        length_tokens_text_only = len(text_only_input_ids)

        line[self.filter_name_char] = length_char
        line[self.filter_name_tokens] = length_token
        line[self.filter_name_tokens_text_only] = length_tokens_text_only
        line[self.filter_name_tokens_trainable_only] = ret["trainable_token_count"]
        line[self.filter_name_images] = length_images

        return line

def get_conversation_length(conv):
    """
    Returns the length of the conversation in characters.
    """
    length = 0
    for msg in conv:
        if msg["value"] is not None:
            length += len(str(msg["value"]))
    return length


def collate_fn(batch):
    return batch

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as file_ptr:
        data = [json.loads(line) for line in file_ptr]
    return data

def make_output_path(args: Namespace):
    assert isinstance(args.input_file, str) \
        and args.input_file.endswith(".jsonl") \
            and args.input_file.count(".jsonl") == 1
    return args.input_file.replace(".jsonl", f"_{args.filter_name}.jsonl")

def main(args):
    output_file = make_output_path(args)

    # time stats
    st = time.time()

    # load data in batched manner
    dataset = LengthFilterDataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # since no inter-device transfers 
        prefetch_factor=args.prefetch_factor,
    )

    total_chars = 0
    total_tokens = 0
    total_tokens_text_only = 0
    total_tokens_trainable_only = 0
    total_images = 0
    total_samples = 0

    with open(output_file, "w") as out_ptr:
        for lines in tqdm(dataloader, total=len(dataloader), ascii=True):
            for line in lines:
                total_chars += line[dataset.filter_name_char]
                total_tokens += line[dataset.filter_name_tokens]
                total_tokens_text_only += line[dataset.filter_name_tokens_text_only]
                total_tokens_trainable_only += line[dataset.filter_name_tokens_trainable_only]
                total_images += line[dataset.filter_name_images]
                total_samples += 1
                out_ptr.write(json.dumps(line) + "\n")

    print(f"Total samples: {total_samples}, Total characters: {total_chars}, Total tokens: {total_tokens}, Total text-only tokens: {total_tokens_text_only}, Total trainable-only tokens: {total_tokens_trainable_only}, Total images: {total_images}")
    print(f"Took {time.time()-st:.2f} seconds to write everything!")

    return args.input_file, output_file, dataset.filter_names

def test(input_file: str, output_file: str, filter_names: list[str], delete_old: bool):
    """
    Use this function as a final "unit-test" for your script (may take some time if your .jsonl is large). 
    :param input_file: path to input jsonl file;  
    :param output_file: path to output jsonl file:
    """
    
    # load input and output .jsonl files
    input_data = load_jsonl(input_file) 
    output_data = load_jsonl(output_file)

    # no lines should be added or removed
    assert len(output_data) == len(input_data)

    # files should be row-wise identical except for the new key
    for row_in, row_out in zip(input_data, output_data):

        # make sure new key is there
        for filter_name in filter_names:
            assert filter_name in row_out

        # make sure other values did not change
        for k, v in row_in.items():
            assert row_out[k] == v, \
                f"Value at key {k} changed from {v} to {row_out[k]}"
            
    print("Test ran through with no assertion errors =)")

    if delete_old:
        print("Deleting the input file {}!!".format(input_file))
        os.remove(input_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True, help="Path to where the images are stored for this dataset.")
    parser.add_argument('--ds_name', type=str, required=True, help="Dataset name.")
    parser.add_argument('--filter_name', type=str, default='length')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--tokenizer_name', type=str, default='OpenGVLab/InternVL2_5-1B')
    parser.add_argument('--conv_style', type=str, default='internvl2_5')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--downsample_ratio', type=float, default=0.5)
    parser.add_argument('--delete_old', action="store_true", help="whether to delete the input_file and only keep the output_file")
    args = parser.parse_args()
    
    input_file, output_file, filter_names = main(args)

    test(input_file, output_file, filter_names, delete_old=args.delete_old)
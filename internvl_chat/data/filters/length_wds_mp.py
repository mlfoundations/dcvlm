import os
import random
import os.path as osp
import multiprocessing as mp
import logging
import numpy as np
import torch

from argparse import ArgumentParser
from argparse import Namespace

from functools import partial
from glob import glob
from tqdm import tqdm
from PIL import Image

from copy import deepcopy
from collections import OrderedDict
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from itertools import islice

import webdataset as wds
from webdataset import TarWriter
from webdataset.tariterators import url_opener, tar_file_expander, group_by_keys
from webdataset.utils import pytorch_worker_info

from constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                       IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                       IMG_START_TOKEN, QUAD_END_TOKEN,
                       QUAD_START_TOKEN, REF_END_TOKEN,
                       REF_START_TOKEN)
from utils.dataloading import (build_transform, dynamic_preprocess, preprocess_internvl2_5)

logger = logging.getLogger(__name__)

##### Copied from: https://github.com/mlfoundations/open_clip/blob/7260a46e7b4bcf518f5200fea06da5bc85aae025/src/open_clip_train/data.py#L180C1-L183C16
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def get_conversation_length(conv):
    """
    Returns the length of the conversation in characters.
    """
    length = 0
    for msg in conv:
        length += len(msg["value"])
    return length

ALLOWED_IMG_EXTS = ['jpg', 'jpeg', 'png', 'webp']

DATASET_FORMAT_ROUTERS = {
    "relaion_2b": "_format_icpairs_from_img_txt",
    "grit_caption": "_format_icpairs_from_img_txt",
    "mint_html": "_format_mmdocs",
    "mint_pdf": "_format_mmdocs",
    "multimodal_textbook": "_format_mmdocs",
    "wanjuan": "_format_mmdocs",
    "asv1": "",  # NOTE: this is slightly different
    "omnicc": "_format_mmdocs",
    "datacomp1b": "_format_icpairs_from_img_txt",
    "recap1b": "",  # NOTE: this is slightly different
    "internvl_sa1b_caption": "_format_icpairs_from_img_txt",
    "facecaption": "_format_icpairs_from_img_txt",
}

#### used for image-caption pairs
LLAVA_DESC_SHORT_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo’s key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]


class LengthFilterWebDataset(IterableDataset):
    """Dataset for iterable loading of large image-text pairs and multimodal documents."""

    def __init__(
        self,
        tarfile,
        template_name,  # conversation format template
        data_type,
        length,
        tokenizer_name,  # tokenizer name
        ds_name,  # dataset name
        num_image_token,  # this is usually going to be 256 by default for InternVL2.5
        filter_name_char="length_chars",
        filter_name_tokens="length_tokens",
        image_size=448,  # this is default image-size used by InternViT vision encoder
        is_train=True,  # mainly used for augmentation
        pad2square=False,  # should images be padded to square images or not?
        dynamic_image_size=False,  # this enables dynamic resolution tiling preprocessing of images as described in the internvl2.5 paper
        use_thumbnail=False,  # whether to add the thumbnail as an image tile or not
        min_dynamic_patch=1,  # minimum number of tiles to split each image into for dynamic preprocessing
        max_dynamic_patch=12,  # maximum number of tiles to split each image into for dynamic preprocessing
        normalize_type='imagenet',
        use_packed_ds=True,
    ):  
        super(LengthFilterWebDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.filter_name_char = filter_name_char
        self.filter_name_tokens = filter_name_tokens
        logger.info(f'[LengthFilterWebDataset] num_image_token: {num_image_token}')
        logger.info(f'[LengthFilterWebDataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[LengthFilterWebDataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[LengthFilterWebDataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.use_packed_ds = use_packed_ds
        self.dataset_type = 'web'
        self.max_num_images = 1  # this is probably not used anywhere else in the code so fine to hard-code it!
        self.max_tokens = self.tokenizer.model_max_length
        # TODO: quick resume
        self._state_dict = {}
        self.iter = 0

        self.data_type = data_type
        self.length = length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        ### Now we want to construct the actual webdataset
        self.dataset = wds.DataPipeline(
            wds.WebDataset(tarfile),
            wds.decode("pilrgb", handler=log_and_continue),  # this decodes the images and json
            wds.map(self.image_format_sample_dicts),  # this function formats handles the dynamic preprocessing for images
            wds.map(self.conversation_format_sample_dicts),  # this function formats the sample dict into a viable one for preprocessing into the current conversation style
            wds.map(self.append_length_values),  # formatter that yields tokenized samples
        ) 

    def get_num_images_from_sample(self, sample):
        c = 0
        for samp_key, value in sample.items():
            if isinstance(value, Image.Image):
                c += 1
        return c

    def pop_image_keys_from_sample(self, sample):
        keys_to_pop = [key for key, value in sample.items() if isinstance(value, Image.Image)]
        # this is a hack to keep the positional information of multimodal sequences while
        # getting rid of the values which we have already persisted in `pixel_values`
        for key in keys_to_pop:
            sample[key] = key
        return sample

    def image_format_sample_dicts(self, sample):
        '''
        This function handles the dynamic preprocessing of images as required by InternVL-2.5

        Input: dictionary from webdataset

        Output: dictionary of webdataset augmented with information about the dynamically preprocessed images
        '''
        sample_copy = deepcopy(sample)
        transform = self.get_transform()

        tiles_to_return, num_tiles_per_image = [], []
        num_image = self.get_num_images_from_sample(sample)

        # We need to ensure that the `pixel_values` 
        # that are fed in for the mmdocs case are 
        # correctly sorted based on the position
        # of the image in the multimodal sequence,
        # hence we sort to be extra sure that order
        # is preserved.
        if self.data_type in ["mmdoc_mt", "mmdoc_ar", "mmdoc_ic"]:
            sample = OrderedDict(sorted(
                sample.items(),
                key=lambda item: int(item[0].split('.')[0]) if item[0].split('.')[0].isdigit() else float('inf')
            ))

            # Here, we are popping the unnecessary json files from img2dataset with image metadata
            keys_to_pop = []
            for k, v in sample.items():
                if k.endswith(".json"):
                    keys_to_pop.append(k)
            for k in keys_to_pop:
                sample.pop(k)

            # If the last item in the sequence is an image, we drop it since we cannot have loss over the last image.
            pop_from_last = []
            for samp_key, value in reversed(sample.items()):
                if not samp_key.split(".")[0].isdigit():
                    continue
                if isinstance(value, Image.Image):
                    pop_from_last.append(samp_key)
                else:
                    break
            for k in pop_from_last:
                sample.pop(k)

        # Iterate over all possible images in the sequence
        for samp_key, value in sample.items():
            # this works because we decode with pilrgb
            if isinstance(value, Image.Image):
                if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                    curr_tiles = dynamic_preprocess(value, min_num=self.min_dynamic_patch,
                                            max_num=max(1, self.max_dynamic_patch // num_image),
                                            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    tiles_to_return += curr_tiles
                    num_tiles_per_image.append(len(curr_tiles))
                else:
                    tiles_to_return.append(value)
                    num_tiles_per_image.append(1)

        del sample

        assert sum(num_tiles_per_image) == len(tiles_to_return), 'Tiles to return and total num tiles per image do not match, sum(num_tiles_per_image): {}, len(tiles_to_return)'.format(sum(num_tiles_per_image), len(tiles_to_return))

        # Get all the transformed tiles
        pixel_values = [transform(tile) for tile in tiles_to_return]
        pixel_values = torch.stack(pixel_values)
        total_num_tiles = pixel_values.size(0)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens_per_image = [self.num_image_token * num_tile for num_tile in num_tiles_per_image]

        # Keep the original sample without modifications
        ret_dict = sample_copy
        
        # Append information needed for tokenization
        ret_dict['pixel_values'] = pixel_values  # this is the processed tensor of all stacked tiles across all images
        ret_dict['num_image_tokens_per_image'] = num_image_tokens_per_image  # this is list of num tokens per image, across images in the sequence
        ret_dict['num_tiles_per_image'] = num_tiles_per_image  # this is a list of number of tiles per image in the sequence
        ret_dict['total_num_tiles'] = total_num_tiles  # this is the total number of tiles across all images
        ret_dict['num_image'] = num_image  # this is the total number of raw images in the sequence which will be used for formatting
        return ret_dict

    def conversation_format_sample_dicts(self, sample):
        '''
        This function will convert the raw sample dict yielded by webdataset
        to a format like {__key__, conversations}

        Input: dictionary from webdataset

        Output: dictionary with conversations
        '''
        filters_dict = sample.get("filters.json", {})

        # this is dataset dependent (different sharded datasets have different sample keys)
        ds_formatter = DATASET_FORMAT_ROUTERS[self.ds_name]
        if ds_formatter == "_format_icpairs_from_img_txt":
            out_ = self._format_icpairs_from_img_txt(sample)
        elif ds_formatter == "_format_mmdocs":
            if self.data_type == "mmdoc_mt":
                out_ = self._multimodal_doc_multi_turn_format(sample)
            elif self.data_type == "mmdoc_ic":
                out_ = self._multimodal_doc_in_context_format(sample)
            elif self.data_type == "mmdoc_ar":
                out_ = self._multimodal_doc_ar_format(sample)
            else:
                raise ValueError("{} formatting not known!".format(self.data_type))
        
        out_["filters.json"] = filters_dict
        return out_
    
    def append_length_values(self, sample):
        # Preprocess the conversation with images to get tokenized output
        preprocess_function = preprocess_internvl2_5
        ret = preprocess_function(template_name=self.template_name, sources=[deepcopy(sample['conversations'])],
                                  tokenizer=self.tokenizer, num_image_token_list=sample['num_image_tokens_per_image'],
                                  group_by_length=False,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=sample['num_image'])


        # Create the final return dictionary
        length_char = get_conversation_length(sample['conversations'])
        length_token = ret['input_ids'].shape[1]

        # Append the length values to the sample
        filters_dict = sample.get("filters.json", {})
        filters_dict[self.filter_name_char] = length_char
        filters_dict[self.filter_name_tokens] = length_token
        sample["filters.json"] = filters_dict

        # Remove keys that were added for processing
        sample.pop("pixel_values", None)
        sample.pop("num_image_tokens_per_image", None)
        sample.pop("num_tiles_per_image", None)
        sample.pop("total_num_tiles", None)
        sample.pop("num_image", None)

        return sample

    def _multimodal_doc_multi_turn_format(self, sample):
        # TODO
        keys_to_pop = []
        for k, v in sample.items():
            if k.endswith(".json"):
                keys_to_pop.append(k)
        for k in keys_to_pop:
            sample.pop(k)

        multimodal_sequence = {k:v for k, v in sample.items() if k.split(".")[0].isdigit()}

        for k, v in multimodal_sequence.items():
            print(k)
        exit()

        return sample

    def _multimodal_doc_in_context_format(self, sample):

        multimodal_sequence = [v for k, v in sample.items() if k.split(".")[0].isdigit()]

        # If the last item in the sequence is an image, we drop it since we cannot have loss over the last image.
        while isinstance(multimodal_sequence[-1], Image.Image):
            multimodal_sequence = multimodal_sequence[:-1]

        assert not isinstance(multimodal_sequence[-1], Image.Image), "Last item is an image"
        assert len(multimodal_sequence) > 1, "Single-item multimodal sequence after popping images, invalid and dropping this...!"

        # Here, we construct the conversation such that everything until the last text part of the sequence is in the human part of the conversation
        # and the last text part is in the gpt part of the conversation (over which we take the loss)
        human_part = ""
        gpt_part = ""
        for mm_index, v in enumerate(multimodal_sequence):
            if mm_index == len(multimodal_sequence) - 1:
                gpt_part = v
            else:
                if isinstance(v, Image.Image):
                    human_part += "<image>{}".format(random.choice(["\n", " ", "\t", "."])) if random.choice([0, 1]) else "<image>"
                else:
                    human_part += "{}{}".format(v, random.choice(["\n", " ", "\t", "."])) if random.choice([0, 1]) else v

            conversations = [{"from": "human", "value": human_part}, {"from": "gpt", "value": gpt_part}]

        sample["conversations"] = conversations
        return sample

    def _multimodal_doc_ar_format(self, sample):
        # TODO
        return sample


    def _format_icpairs_from_img_txt(self, sample):
        '''
        Formatter for image-text pairs from img and txt keys.

        Here, we assume sample contains 'txt' and 'jpg'/'png'/'jpeg'/'webp' keys for text caption and images.

        Input: sample dict containing keys with texts and images

        Returns: conversation dictionary 
        '''
        image_beginning = random.choice([0, 1])
        if image_beginning:
            conversations = [{"from": "human", "value": "<image>\n{}".format(random.choice(LLAVA_DESC_SHORT_PROMPTS))}, {"from": "gpt", "value": sample['txt']}]
        else:
            conversations = [{"from": "human", "value": "{}\n<image>".format(random.choice(LLAVA_DESC_SHORT_PROMPTS))}, {"from": "gpt", "value": sample['txt']}]

        sample["conversations"] = conversations
        return sample

    def _format_mmdocs(self, sample):
        pass

    def _format_grounding(self, sample):
        pass

    def __len__(self):
        return self.length


    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def __iter__(self):
        yield from self.dataset


def log_and_continue(exn):
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def make_output_folder(args: Namespace):
    assert isinstance(args.input_folder, str) and osp.exists(args.input_folder) and osp.isdir(args.input_folder)
    output_folder = osp.join(args.input_folder, args.filter_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def seek_key(iterator, key):
    while True:
        try:
            sample = next(iterator)
            assert sample['__key__'] == key
            break
        except AssertionError:
            continue
    return sample


def test(input_folder: str, output_folder: str, filter_names: list[str]):
    print("Running tests (may take a while...)")
    
    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)
    random.shuffle(tarfiles)

    # run the test on a single random tar
    input_tarfile = osp.join(input_folder, osp.basename(tarfiles[0]))
    output_tarfile = osp.join(output_folder, osp.basename(tarfiles[0]))
    assert osp.exists(output_tarfile) and osp.isfile(output_tarfile)

    # create corresponding datasets and iterators
    input_dataset = wds.WebDataset(input_tarfile).decode("pil", handler=log_and_continue)
    output_dataset = wds.WebDataset(output_tarfile).decode("pil", handler=log_and_continue)
    output_iter = iter(output_dataset)

    for inp_sample in tqdm(input_dataset):
        # some samples in the original wds may be corrupted / non-decodable
        # those will not have a match in the new output shards
        sample_key = inp_sample['__key__']
        out_sample = seek_key(output_iter, key=sample_key)

        # check filters are there
        assert "filters.json" in out_sample, \
            f"There is no filters.json key in the sample {sample_key}."
        
        # check the new key is there
        filters = out_sample["filters.json"]
        for filter_name in filter_names:
            assert filter_name in filters, \
                f"The filters.json dict of sample {sample_key} does not have a {filter_name} entry."

        # check samples are identical 
        for k, v in inp_sample.items():
            assert k in out_sample, f"Did you remove key {k} from the output samples?"

            # local keys which we don't care about
            if k.startswith("_"): continue

            # pHash for images
            if isinstance(v, Image.Image):
                tol = 5
                in_hash = imagehash.phash(v, hash_size=10)
                out_hash = imagehash.phash(out_sample[k], hash_size=10)
                assert in_hash - out_hash <= tol, f"Images at key {sample_key} ({k}) look different."

            # this will mostly do string / annotation comparison
            elif k != "filters.json":
                assert out_sample[k] == v, f"{k} is wrongly matched"
            
            # this checks the "filters.json" is unchanged except for :param filter_name:
            # (i.e., any other previous filter-key therein remained unchanged)
            else:
                input_filters_dict = inp_sample[k]
                for k1, v1 in input_filters_dict.items():
                    assert k1 in filters
                    assert filters[k1] == v1

    print("Tests ran through with no Assertion Errors =)")


def score_and_dump_tarfile(tarfile: str, args, output_folder: str):
    num_image_token = int((args.image_size // args.patch_size) ** 2 * (args.downsample_ratio ** 2))
    # Init a WebDataset with the current tar
    encoded_dataset = wds.WebDataset(tarfile)
    decoded_dataset = LengthFilterWebDataset(
        tarfile,
        args.template_name,
        args.data_type,
        None,
        args.tokenizer_name,
        args.ds_name,
        num_image_token,
        args.filter_name_char,
        args.filter_name_tokens
    )
    encoded_iter = iter(encoded_dataset)
    
    # programmatically create output tar
    output_tar = osp.join(output_folder, osp.basename(tarfile))
    print(f"Will write to {output_tar}")
    with TarWriter(output_tar) as writer:
        
        # iterate through the samples in the decoded WebDataset 
        for dec_sample in tqdm(decoded_dataset, desc="Appending length values.", ascii=True):
            # seek the corresponding encoded sample
            enc_sample = seek_key(encoded_iter, key=dec_sample['__key__'])
            
            # Length was already appended by the Dataset - nothing to do here
            filters_dict = dec_sample.get("filters.json", {})
            
            # write the encoded sample on disk (storage-friendly)
            enc_sample["filters.json"] = filters_dict
            writer.write(enc_sample)

    return output_tar


def main(args):

    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)

    # parent folder to dump new tarfiles
    output_folder = make_output_folder(args)

    # dump all new tarfiles with injected scores (multiprocessing version)
    # NOTE: the scoring and dumping logic will be in the `score_and_dump_tarfile` function now
    func_for_multiprocessing = partial(score_and_dump_tarfile, args=args, output_folder=output_folder)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        
        # imap_unordered is preferred over imap and map
        output_tarfiles = pool.imap_unordered(
            func=func_for_multiprocessing, iterable=tarfiles
        )

        # quick check that things went smoothly
        for idx, output_tarfile in enumerate(output_tarfiles): 
            assert osp.exists(output_tarfile) and osp.isfile(output_tarfile)
            print(f"[{idx+1} / {len(tarfiles)}]\t {output_tarfile} dumped!")
    
    # return paths for testing
    return args.input_folder, output_folder


if __name__ == "__main__":
    # very simple cmd-line args, feel free to extend
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--filter_name', type=str, default='length')
    parser.add_argument('--template_name', type=str, default='internvl2_5')
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--data_type', type=str, default='mmdoc_ic')
    parser.add_argument('--prefetch_factor', type=int, default=6)
    parser.add_argument('--tokenizer_name', type=str, default='OpenGVLab/InternVL2_5-1B')
    parser.add_argument('--meta_path', type=str, default='meta.json')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--downsample_ratio', type=float, default=0.5)
    args = parser.parse_args()

    args.filter_name_char = args.filter_name + "_chars"
    args.filter_name_tokens = args.filter_name + "_tokens"
    args.filter_names = [args.filter_name_char, args.filter_name_tokens]
    
    # do your filtering here
    input_folder, output_folder = main(args)

    # and run a simple unit-test
    # NOTE: this may not cover all cases, especially sample-wise key checks
    # but we can take it as a reference / starting point
    test(input_folder, output_folder, args.filter_names)

import logging
import random
from copy import deepcopy
from typing import Dict, Literal, Optional
from collections import OrderedDict

import numpy as np

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist

try:
    from constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                        IMG_START_TOKEN, QUAD_END_TOKEN,
                                        QUAD_START_TOKEN, REF_END_TOKEN,
                                        REF_START_TOKEN)
except ModuleNotFoundError as e:
    from .constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                        IMG_START_TOKEN, QUAD_END_TOKEN,
                                        QUAD_START_TOKEN, REF_END_TOKEN,
                                        REF_START_TOKEN)

try:
    from dataloading import dynamic_preprocess
except ModuleNotFoundError as e:
    from .dataloading import dynamic_preprocess

try:
    from dataloading import preprocess_internvl2_5
except ModuleNotFoundError as e:
    from .dataloading import preprocess_internvl2_5

try:
    from dataloading import build_transform
except ModuleNotFoundError as e:
    from .dataloading import build_transform

from PIL import Image
from torch.utils.data import IterableDataset
import torch.utils.data

from itertools import islice
import webdataset as wds
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample, group_by_keys
from webdataset.filters import pipelinefilter
from webdataset.utils import pytorch_worker_info

logger = logging.getLogger(__name__)

##### Copied from: https://github.com/mlfoundations/open_clip/blob/7260a46e7b4bcf518f5200fea06da5bc85aae025/src/open_clip_train/data.py#L180C1-L183C16
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def tarfile_to_samples_throw_wrapped(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys(files, handler=handler)
    return samples

###### Copied from: https://github.com/webdataset/webdataset/blob/75bf1455d60cc8bcb2081a6a7b4a3b561f405a3a/webdataset/shardlists.py#L101
def split_by_worker_reimp(src):
    """Split the input sequence by PyTorch DataLoader worker.

    Args:
        src: The input sequence to be split.

    Yields:
        Elements from the input sequence based on the worker's ID.
    """
    rank, world_size, worker, num_workers = pytorch_worker_info()

    if num_workers > 1:
        yield from islice(src, worker, None, num_workers)
    else:
        yield from src

###### Copied from: https://github.com/webdataset/webdataset/blob/75bf1455d60cc8bcb2081a6a7b4a3b561f405a3a/webdataset/shardlists.py#L63
def split_by_node_reimp(src, group=None):
    """Split the input sequence by PyTorch distributed rank.

    Args:
        src: The input sequence to be split.
        group: The process group for distributed training.

    Yields:
        Elements from the input sequence based on the node's rank.
    """
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)

    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src

##### copied from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/data.py#L236-239
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
_SHARD_SHUFFLE_SIZE = 1000

ALLOWED_IMG_EXTS = ['jpg', 'jpeg', 'png', 'webp']

DATASET_FORMAT_ROUTERS = {
    "relaion_2b": "_format_icpairs_from_img_txt",
    "grit_caption": "_format_icpairs_from_img_txt",
    "mint_html": "_format_mmdocs",
    "mint_pdf": "_format_mmdocs",
    "multimodal_textbook": "_format_mmdocs",
    "wanjuan": "_format_mmdocs",
    "asv1": "_format_icpairs_from_img_txt",  ### NOTE (Vishaal, 15/05): For now, we are just going to use the image-caption pairs format for this since we don't have the grounding dataset impl
    "omnicc": "_format_mmdocs",
    "datacomp1b": "_format_icpairs_from_img_txt",
    "recap1b": "",  # NOTE: this is slightly different
    "internvl_sa1b_caption": "_format_icpairs_from_img_txt_internvl_sa_1b",  # this is slightly weird, it has .txt keys instead of txt keys, hence added a separate function to avoid confusion.
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

def collate_(batch):    
    return batch

def get_conversation_length(conv):
    """
    Returns the length of the conversation in characters.
    """
    length = 0
    for msg in conv:
        length += len(msg["value"])
    return length

#### NOTE (Vishaal, 15/05): this class is a copy of the WrappedWebDataset class with some added utils for adding length info
class LengthFilterWebDataset(IterableDataset):
    """Dataset for iterable loading of large image-text pairs and multimodal documents."""

    def __init__(
        self,
        template_name,  # conversation format template
        meta,  # json file containing metadata of dataset --> this will be a dict for us!
        tokenizer,
        ds_name,  # dataset name
        num_image_token,  # this is usually going to be 256 by default for InternVL2.5
        image_size=448,  # this is default image-size used by InternViT vision encoder
        is_train=False,  # mainly used for augmentation  ## NOTE (Vishaal, 15/05): patched this to not do any augmentation during computing the token stats
        pad2square=False,  # should images be padded to square images or not?
        dynamic_image_size=True,  # this enables dynamic resolution tiling preprocessing of images as described in the internvl2.5 paper  ## NOTE (Vishaal, 15/05): for computing token stats, lets always use dynamic image preprocessing
        use_thumbnail=False,  # whether to add the thumbnail as an image tile or not
        min_dynamic_patch=1,  # minimum number of tiles to split each image into for dynamic preprocessing
        max_dynamic_patch=12,  # maximum number of tiles to split each image into for dynamic preprocessing
        repeat_time=1,  # sampling frequency
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=True,  # we set this to true so that we don't do any padding later on in the preprocess function!
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,  # in the calling function, this is always the same as `use_packed_ds`
        force_shuffle=False,  # in the calling function, this is always the same as `use_packed_ds`
        random_seed=0,  # this is always set as the id of the dataset in the calling function
        batch_size=128,  # we use this to do batching in the webdataset pipeline directly!
    ):
        super(LengthFilterWebDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.batch_size = batch_size
        logger.info(f'[LengthFilterWebDataset] num_image_token: {num_image_token}')
        logger.info(f'[LengthFilterWebDataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[LengthFilterWebDataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[LengthFilterWebDataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        assert self.template_name == "internvl2_5", "For the length filter, we have to use the internvl2_5 preprocessing function, please check again!"

        assert self.num_image_token == 256, "By default, the number of tokens per image-tile should be 256, please check again!"

        self.image_size = image_size

        assert self.image_size == 448, "By default, InternViT that we use for training uses 448x448 as input image-size, please check again!"

        self.is_train = is_train
        self.pad2square = pad2square

        ### NOTE: (Vishaal, 15/05) --> these are the token stats that we will accumulate
        self.filter_name_char = "length_chars"
        self.filter_name_tokens = "length_tokens"
        self.filter_name_tokens_text_only = "length_tokens_text_only"
        self.filter_name_tokens_trainable_only = "length_tokens_trainable_only"
        self.filter_name_images = "length_images"
        self.filter_names = [self.filter_name_char, self.filter_name_tokens, self.filter_name_tokens_text_only, self.filter_name_tokens_trainable_only, self.filter_name_images]

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'web'
        self.max_num_images = 1  # this is probably not used anywhere else in the code so fine to hard-code it!
        self.max_tokens = tokenizer.model_max_length
        # TODO: quick resume
        self._state_dict = {}

        assert 'annotation' not in meta, f'annotation exists are you sure this is a webdataset?'
        # NOTE (Vishaal, 15/05): we don't need this for the length filter, we just iterate to the end!
        # assert 'length' in meta and meta['length'] > 0, f'we need length metadata for the webdataset, currently length either does not exist or is 0!'

        self.rng = np.random.default_rng(seed=random_seed)

        self.root = meta['root']
        self.data_type = meta.get('data_type', "captioning")  # by default, assume all webdatasets are image-text pair datasets
        # NOTE (Vishaal, 15/05): we don't need this for the length filter, we just iterate to the end!
        # self.length = int(meta['length'] * meta['repeat_time'])
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        ### Now we want to construct the actual webdataset
        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(self.root),  # this will give an iterator over all tars
            # wds.shuffle(_SHARD_SHUFFLE_SIZE),  # this will shuffle the tars  ### NOTE (Vishaal, 15/05) --> we don't shuffle for the length filter!
            # split_by_node_reimp,  # this will distribute across all the ranks (num_gpus * num_nodes)  ### NOTE (Vishaal, 15/05) --> we also don't want to split across gpu-ranks, the length filter will all run on cpu jobs!
            split_by_worker_reimp,  # this will distribute across all cpu workers within a rank (num_workers per gpu)
            tarfile_to_samples_throw_wrapped,  # this will give us sample dicts per worker
            # wds.shuffle(
            #     bufsize=_SAMPLE_SHUFFLE_SIZE,
            #     initial=_SAMPLE_SHUFFLE_INITIAL,
            # ),  # this will shuffle samples in memory, per worker  ### NOTE (Vishaal, 15/05) --> again, no shuffling for length filter!
            wds.decode("pilrgb", handler=log_and_continue),  # this decodes the images and json
            wds.map(self.image_format_sample_dicts, handler=log_and_continue),  # this function formats handles the dynamic preprocessing for images
            wds.map(self.conversation_format_sample_dicts, handler=log_and_continue),  # this function formats the sample dict into a viable one for preprocessing into the current conversation style
            # NOTE: we're currently relying on InternVL2.5, this may change in the future
            wds.map(self.final_format_for_forward, handler=log_and_continue),  # final formatter that yields samples that can be used directly in HF-trainer
            # (Optional) We may have some batching here, which is the recommended way to 
            # operate with wds rather than batching in the loader, so something like
            wds.batched(self.batch_size, collation_fn=collate_)
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

        transform = self.get_transform()

        tiles_to_return, num_tiles_per_image = [], []

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

            # Here, we are popping the unnecessary json files from img2dataset with image metadata and other auxiliary files added by our filters (e.g. sscd metadata)
            keys_to_pop = []
            for k, v in sample.items():
                ##### NOTE (Vishaal, 15/05): we have to keep the filters.json so that it can be accessed later on for adding the length metadata!
                if k == "filters.json":
                    continue
                if k.endswith(".json"):  # this is to drop metadata added by img2dataset
                    keys_to_pop.append(k)
                if k.endswith(".phash.txt"):  # this is to drop the phash value of the image stored by our image descriptor scripts
                    keys_to_pop.append(k)
                if k.endswith(".sha256.txt"):  # this is to drop the sha256 value of the image bytes stored by our image descriptor scripts
                    keys_to_pop.append(k)
                if k.endswith(".sscd_disc_mixup.pt"):  # this is to drop the sscd descriptor embedding of the image stored by our image descriptor scripts
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

        ### NOTE (15/05): we move this here to ensure that we get the correct number of images after the cleaning step for multimodal documents
        num_image = self.get_num_images_from_sample(sample)

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

        assert sum(num_tiles_per_image) == len(tiles_to_return), 'Tiles to return and total num tiles per image do not match, sum(num_tiles_per_image): {}, len(tiles_to_return)'.format(sum(num_tiles_per_image), len(tiles_to_return))

        # Get all the transformed tiles
        pixel_values = [transform(tile) for tile in tiles_to_return]
        pixel_values = torch.stack(pixel_values)
        total_num_tiles = pixel_values.size(0)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens_per_image = [self.num_image_token * num_tile for num_tile in num_tiles_per_image]

        ret_dict = sample
        # ret_dict = self.pop_image_keys_from_sample(ret_dict)  # NOTE: uncomment this for efficiency
        ret_dict['pixel_values'] = pixel_values  # this is the processed tensor of all stacked tiles across all images
        ret_dict['num_image_tokens_per_image'] = num_image_tokens_per_image  # this is list of num tokens per image, across images in the sequence
        ret_dict['num_tiles_per_image'] = num_tiles_per_image  # this is a list of number of tiles per image in the sequence
        ret_dict['total_num_tiles'] = total_num_tiles  # this is the total number of tiles across all images
        ret_dict['num_image'] = num_image  # this is the total number of raw images in the sequence which will be used for formatting
        return ret_dict

    def final_format_for_forward(self, sample):

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        ret = preprocess_function(template_name=self.template_name, sources=[deepcopy(sample['conversations'])],
                                  tokenizer=self.tokenizer, num_image_token_list=sample['num_image_tokens_per_image'],
                                  group_by_length=False,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=sample['num_image'])

        # Create the final return dictionary
        length_char = get_conversation_length(sample['conversations'])
        length_token = ret['input_ids'].shape[1]
        length_token_text_only = ret['text_only_input_ids'].shape[1]
        length_token_trainable_only = ret['trainable_token_count']
        length_images = sample['num_image']

        # Append the length values to the sample
        filters_dict = sample.get("filters.json", {})
        filters_dict[self.filter_name_char] = length_char
        filters_dict[self.filter_name_tokens] = length_token
        filters_dict[self.filter_name_tokens_text_only] = length_token_text_only
        filters_dict[self.filter_name_tokens_trainable_only] = length_token_trainable_only
        filters_dict[self.filter_name_images] = length_images
        sample["filters.json"] = filters_dict

        ### NOTE (Vishaal, 15/05) --> we do not need any of this stuff for length filter!
        # # Calculate position_ids for packed dataset
        # position_ids = ret['attention_mask'].long().cumsum(dim=-1) - 1
        # position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        # image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        # check_ = (ret['input_ids'][0] == image_end_token_id).sum()
        # assert check_ == num_img, f'image tokens are truncated, there are {check_} image-end-tokens, should have been {num_img}, this dataset is {self.ds_name}'

        ### NOTE (Vishaal, 15/05) --> we can directly return the sample since it has all the token stats we need!
        # # Create the final return dictionary
        # ret = dict(
        #     input_ids=ret['input_ids'][0],
        #     labels=ret['labels'][0],
        #     attention_mask=ret['attention_mask'][0],
        #     position_ids=position_ids[0],
        #     pixel_values=pv_,
        #     image_flags=torch.tensor([1] * tot_num_tiles, dtype=torch.long)
        # )

        # return ret

        # Remove keys that were added for processing
        sample.pop("pixel_values", None)
        sample.pop("num_image_tokens_per_image", None)
        sample.pop("num_tiles_per_image", None)
        sample.pop("total_num_tiles", None)
        sample.pop("num_image", None)

        return sample

    def conversation_format_sample_dicts(self, sample):
        '''
        This function will convert the raw sample dict yielded by webdataset
        to a format like {__key__, conversations}

        Input: dictionary from webdataset

        Output: dictionary with conversations
        '''
        # this is dataset dependent (different sharded datasets have different sample keys)
        ds_formatter = DATASET_FORMAT_ROUTERS[self.ds_name]
        if ds_formatter == "_format_icpairs_from_img_txt":
            out_ = self._format_icpairs_from_img_txt(sample)
        elif ds_formatter == "_format_icpairs_from_img_txt_internvl_sa_1b":
            out_ = self._format_icpairs_from_img_txt_internvl_sa_1b(sample)
        elif ds_formatter == "_format_mmdocs":
            if self.data_type == "mmdoc_mt":
                out_ = self._multimodal_doc_multi_turn_format(sample)
            elif self.data_type == "mmdoc_ic":
                out_ = self._multimodal_doc_in_context_format(sample)
            elif self.data_type == "mmdoc_ar":
                out_ = self._multimodal_doc_ar_format(sample)
            else:
                raise ValueError("{} formatting not known!".format(self.data_type))
        return out_

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

    def _format_icpairs_from_img_txt_internvl_sa_1b(self, sample):
        '''
        Formatter for image-text pairs from img and .txt keys.

        Here, we assume sample contains '.txt' and '.jpg'/'.png'/'.jpeg'/'.webp' keys for text caption and images.

        Input: sample dict containing keys with texts and images

        Returns: conversation dictionary 
        '''
        image_beginning = random.choice([0, 1])
        if image_beginning:
            conversations = [{"from": "human", "value": "<image>\n{}".format(random.choice(LLAVA_DESC_SHORT_PROMPTS))}, {"from": "gpt", "value": sample['.txt']}]
        else:
            conversations = [{"from": "human", "value": "{}\n<image>".format(random.choice(LLAVA_DESC_SHORT_PROMPTS))}, {"from": "gpt", "value": sample['.txt']}]

        sample["conversations"] = conversations
        return sample

    def _format_grounding(self, sample):
        pass

    ## NOTE (Vishaal, 15/05): we will never use the length call of this dataset for the length filter, so we should be good not overriding it!
    # def __len__(self):
    #     return self.length

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        return preprocess_internvl2_5

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def __iter__(self):
        yield from self.dataset
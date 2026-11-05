import os
import json
import random
import torch
import traceback
import numpy as np

from typing import Dict
from copy import deepcopy
from torch.utils.data import Dataset
from PIL import (
    Image, 
    ImageFile, 
    UnidentifiedImageError, 
    PngImagePlugin
)

from internvl.train.constants import IMG_END_TOKEN
from internvl.train.dataset import (
    build_transform,
    dynamic_preprocess, preprocess,
    preprocess_internlm,
    preprocess_internvl2_5, 
    preprocess_mpt,
    preprocess_phi3
)

# Set constants for image processing and logging
IGNORE_INDEX = -100  # this is the token index for ignoring loss. so this would be used for the "human" part of the conversations & image tokens
Image.MAX_IMAGE_PIXELS = 30_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

# get root logger from main
import logging
logger = logging.getLogger()


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,  # conversation format template
        meta,  # json file containing metadata of dataset
        tokenizer,
        tcs_loader,
        ds_name,  # dataset name
        num_image_token,  # this is usually going to be 256 by default for InternVL2.5
        image_size=448,  # this is default image-size used by InternViT vision encoder
        is_train=True,  # mainly used for augmentation
        pad2square=False,  # should images be padded to square images or not?
        group_by_length=False,  # only used if packing is disabled
        dynamic_image_size=False,  # this enables dynamic resolution tiling preprocessing of images as described in the internvl2.5 paper
        use_thumbnail=False,  # whether to add the thumbnail as an image tile or not
        min_dynamic_patch=1,  # minimum number of tiles to split each image into for dynamic preprocessing
        max_dynamic_patch=12,  # maximum number of tiles to split each image into for dynamic preprocessing
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,  # sampling frequency
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,  # in the calling function, this is always the same as `use_packed_ds`
        force_shuffle=False,  # in the calling function, this is always the same as `use_packed_ds`
        random_seed=0,  # this is always set as the id of the dataset in the calling function
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame  # for video data, we probably won't use this!
        self.min_num_frame = min_num_frame  # for video data, we probably won't use this!
        self.sampling_method = sampling_method  # for video data, we probably won't use this!

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1  # this is probably not used anywhere else in the code so fine to hard-code it!
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}
        self.repeat_time = repeat_time

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        total_ranks = torch.distributed.get_world_size()  # NOTE: this is technically pointless, we can reuse `data_world_size`
        self.total_ranks = total_ranks
        current_rank = torch.distributed.get_rank()  # NOTE: this is also technically is the same as `data_rank`

        """
        This section of the code is used to read hundreds of millions of data entries.
        By using caching and splitting the data according to rank, it ensures fast reading
        speed and prevents out-of-memory.
        """
        # Create a cache directory path
        # basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
        # data_dir = os.path.join(os.path.dirname(meta['annotation']), f'{basename}_temp')
        # os.makedirs(data_dir, exist_ok=True)  # Create the cache directory if it does not exist
        
        # # Create a temporary path for the current rank
        # temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.jsonl')

        # Check if the temporary file for the current rank already exists
        # if os.path.exists(temp_path):
        #     # If it exists, read the raw data from the file
        #     with open(temp_path, 'r') as f:
        #         self.raw_data = f.readlines()
        # else:
        #     # If it does not exist, read the raw data from the original annotation file
        #     with open(meta['annotation'], 'r') as f:
        #         self.raw_data = f.readlines()

        #     # Adjust the raw data based on the repeat_time parameter
        #     if repeat_time < 1:
        #         self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        #     else:
        #         # NOTE: the original code (commented below) is likely incorrect:
        #         # this should not be done at the broadcasted list level but at 
        #         # the sample level, because fractional repeat_times greater than 
        #         # 1 won't work!
        #         # we have fixed it with updated code on 26/03
        #         # self.raw_data = self.raw_data * int(repeat_time)
        #         self.raw_data = self.raw_data * int(repeat_time) + self.raw_data[ : int( len(self.raw_data) * (repeat_time - int(repeat_time)) ) ]

        
        # NOTE: (Matteo), we are moving towards a FileStream only implementation of LazySupervisedDataset
        # this means that we simply keep a file pointer in memory and load jsonl entries on the fly
        self.annotation = meta['annotation']

        # NOTE (25/05, Vishaal): after discussing with Matteo, we decided we probably don't need 
        # to do this expensive read and iterate operation, we can simply read the length from 
        # the meta json file. so let's try to do that! 
        # self.stream = open(self.annotation, "r", encoding="utf-8")
        # total_lines = 0
        # for _ in self.stream: total_lines +=1
        # self.stream.close()
        # del self.stream
        total_lines = meta['length']
        
        logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
        lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
        lines_per_rank = max(1, lines_per_rank)

        # Calculate the start and end line numbers for the current rank
        self.start_line = lines_per_rank * current_rank  # Starting line for the current rank
        self.end_line = self.start_line + lines_per_rank  # Ending line for the current rank


        self.rng = np.random.default_rng(seed=random_seed)
        # if self.force_shuffle:
        #     self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        assert not group_by_length
        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        # NOTE: with FileStream we do not have "raw_data" anymore, but we wanna have 
        # a notion of "length" to assign a weight to each dataset anyways withe current 
        # implementation (will be removed soon)
        
        # leaving the old core here for now
        # if not self.use_packed_ds:
        #     return len(self.raw_data) * self.total_ranks
        # else:
        #     return len(self.raw_data)
        
        # this is "len(self.raw_data)"
        rows_per_current_rank = self.end_line - self.start_line

        # still a bit unclear why they do it this way, is it because sampling is done at the
        # rank level before creating a batch? If so, why do they care about scaling by a constant?
        if not self.use_packed_ds:
            return int(rows_per_current_rank * self.total_ranks * self.repeat_time)
        return int(rows_per_current_rank * self.repeat_time)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)  # number of tiles including the thumbnail
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(template_name=self.template_name, sources=[deepcopy(data_item['conversations'])],
                                  tokenizer=self.tokenizer, num_image_token_list=[self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        check_ = (ret['input_ids'][0] == image_end_token_id).sum()
        assert check_ == 1, f'image tokens are truncated, there are {check_} image-end-tokens, should have been 1, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        check_ = (ret['input_ids'][0] == image_end_token_id).sum()
        assert check_ == num_image, f'image tokens are truncated, there are {check_} image-end-tokens, should have been {num_image}, this dataset is {self.ds_name}'
        # assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            # num_worker_per_rank = self.num_workers // self.total_ranks
            # self.raw_data = self.raw_data[self.worker_id % num_worker_per_rank::num_worker_per_rank]
            # logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

            # this will match "dataloader_num_workers" 
            self.num_worker_per_rank = self.num_workers // self.total_ranks
            self.local_start_idx = self.worker_id % self.num_worker_per_rank


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}, try number: {try_cnt}/{max_try}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}, try number: {try_cnt}/{max_try}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}, try number: {try_cnt}/{max_try}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret
    

    def __readline__(self, line: str) -> Dict[str, torch.Tensor]:
        # identical to __getitem__, but loads :param line: on the fly instead of accessing
        # data in memory (we may reformat __getitem__ as well, but nvm)
        try:
            data_item = json.loads(line)
            if 'image' in data_item and len(data_item['image']) != 0:
                if type(data_item['image']) == list:
                    ret = self.multi_modal_multi_image_get_item(data_item)
                else:
                    ret = self.multi_modal_get_item(data_item)
            elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                ret = self.video_get_item(data_item)
            else:
                ret = self.pure_text_get_item(data_item)
        except Exception as e:
            ret = None
            print(e, self.ds_name, flush=True)
            if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                traceback.print_exc()
            try:
                data_item = json.loads(line)
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}.')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}.')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}.')
            except:
                pass
        
        return ret


    def __iter__(self):
        # self._enable_worker_distributed()
        # start_idx = 0

        # assert self.worker_state_key is not None
        # if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
        #     start_idx = self._state_dict[self.worker_state_key]['current_idx']

        #     self._state_dict.pop(self.worker_state_key)

        # if self.worker_id == 0:
        #     logger.info(
        #         f'[{self.ds_name}] [Worker id {self.worker_id}] '
        #         f'begin to iter with {start_idx=}'
        #     )

        # called only at the very beginning before entering the loop (we're yielding)
        stream = open(self.annotation, "r", encoding="utf-8")
        winfo = torch.utils.data.get_worker_info()
        local_worker_id, num_workers_per_rank = winfo.id, winfo.num_workers
        start_offset = self.start_line + local_worker_id

        # little modification to range to enable file-streaming
        for i, line in enumerate(stream):

            # we're out of range for the current rank
            if i < start_offset: continue
            if i >= self.end_line: break

            # we don't wanna read lines of other workers
            local_offset = i - start_offset
            if not local_offset % num_workers_per_rank == 0: continue

            # if we get here, we can try to load
            data = self.__readline__(line)
            if data is None: continue # in-line "error-handling" 

            # this is where we yield for real!!
            yield data

        raise StopIteration
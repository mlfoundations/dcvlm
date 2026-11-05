# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
import os
import os.path as osp
import random
import sys
import wandb
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional
from torch.distributed.elastic.multiprocessing.errors import record

import numpy as np

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler,
                            make_transformers_compatible, fix_checkpoint_config_mismatch)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, MMDOC_FORMATS)
from internvl.train.scale_configs import SCALE_CONFIGS
from internvl.train.dataset import ConcatDataset, TCSLoader, WeightedConcatDataset
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from internvl.train.dataset_packed_lazy import LazyPackedDataset, lazy_packed_collate_fn
from internvl.train.dataset_web import WrappedWebDataset
from internvl.train.dataset_lazy import LazySupervisedDataset
from internvl.train.callbacks import CheckpointCallback, ThroughputLogCallback
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False


# Set constants for image processing and logging
IGNORE_INDEX = -100  # this is the token index for ignoring loss. so this would be used for the "human" part of the conversations & image tokens
Image.MAX_IMAGE_PIXELS = 30_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    model_root: Optional[str] = field(
        default=None,
        metadata={'help': 'Root directory for the models. Used to load models from local paths. This typically is the OFFLINE_MODEL_DIR environment variable.'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    lazy_packing: bool = field(
        default=False,
        metadata={'help': 'Whether to use lazy packing for the dataset. Default is False.'}
    )
    collate_dtypes: str = field(
        default="",
        metadata={'help': "Comma-separated list of data-types to collate into a single entry."}
    )
    mmdoc_format: str = field(
        default="mt",
        metadata={'help': "Formatting style for Multimodal Documents. Acceptable formats are 'ic' (in-context) or 'mt' (multi-turn)."}
    )
    lazy_allow_distributed_replacement: bool = field(
        default=True,
        metadata={'help': 'Whether to only use ResampledShards (recommended), or slice across workers.'}
    )
    lazy_cleanup_freq: Optional[int] = field(
        default=999999,
        metadata={'help': 'When using Lazy Packing, cleanup data pipeline every :lazy_cleanup_freq: iterations.'}
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )
    scale: str = field(
        default=None,
        metadata={'help': 'The scale of the DCVLM benchmark. Options are: small, medium, large, xlarge.'},
    )
    estimated_tokens: Optional[int] = field(
        default=0,
        metadata={'help': 'The estimated number of tokens for a packed sequence. Default is 0 (i.e., unused).'},
    )
    calibration_batches: int = field(
        default=4096,
        metadata={'help': 'Total number of batches (across all ranks) used to estimate avg tokens per packed sequence when estimated_tokens=0 and scale is set. Default is 4096.'},
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=2,
        metadata={'help': 'The number of batches to prefetch. Default is 2.'}
    )


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
    checkpoint=None
):
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()

    if data_args.use_packed_ds and data_args.lazy_packing:
        data_args.num_image_token = model.num_image_token
        collate_dtypes = [
            dtype for dtype in data_args.collate_dtypes.split(",") 
            if dtype
        ]
        return LazyPackedDataset(
            data_args,
            tokenizer,
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            allow_distributed_replacement=data_args.lazy_allow_distributed_replacement,
            log_freq=data_args.log_freq,
            cleanup_freq=data_args.lazy_cleanup_freq,
            seed=data_args.seed,
            checkpoint=checkpoint,
            collate_dtypes=collate_dtypes,
            mmdoc_format=data_args.mmdoc_format
        )

    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())  # this will give all the dicts per dataset from the meta json file
    logger.info(f'Collecting datasets: {list(ds_collections.keys())}')
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']  # this is the repeat factor we set for each dataset that determines the data mixture ratios
        
        ### this overrides the default number of tiles used per image in the dataset,
        ### for now we dont touch this and only set it from the launcher (default=12)
        ### NOTE: maybe worth ablating with higher max_dynamic_path for some domains like OCR?
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        
        # this is the dataset class that we will use for different datasets
        # lazy means we use LazySupervisedDataset --- this is for standard datasets that use the jsonl format
        # webdataset means we use WrappedWebDataset --- this is for the datasets that are stored in the webdataset format as tars
        dataloading_class = ds_collections[ds_name].get("dataloading_class", "webdataset")
        if dataloading_class != "webdataset":
            raise ValueError(
                f"Unknown dataloading_class: {dataloading_class} for dataset: {ds_name}"
            )

        dataset = WrappedWebDataset(
            data_args.conv_style, 
            ds_collections[ds_name],
            tokenizer,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            pad_data=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )

        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)  # add current dataset item to the list of all datasets
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))  # square root sampling was also used in molmo: https://arxiv.org/abs/2409.17146 sec 4
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)  # this is the sum of all datasets that we have
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)

def estimate_avg_tokens(dataloader, num_steps):
    """Iterate `num_steps` batches and return the average non-padding tokens per packed sequence,
    averaged across all distributed ranks."""
    from tqdm import tqdm
    iterator = enumerate(dataloader)
    if dist.get_rank() == 0:
        iterator = tqdm(iterator, desc="Estimating avg pack length...", total=num_steps)
    cum_tokens = 0
    for idx, batch in iterator:
        if idx == num_steps:
            break
        _, num_padding_tokens, _ = batch['statistics'].tolist()
        _, seqlen = batch['input_ids'].size()
        cum_tokens += seqlen - num_padding_tokens
    avg_tokens = torch.tensor([cum_tokens / num_steps], device='cuda')
    dist.barrier()
    dist.all_reduce(avg_tokens, op=dist.ReduceOp.SUM)
    avg_tokens = avg_tokens / dist.get_world_size()
    return int(avg_tokens.item())


@record
def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # make sure packing is setup correctly
    if data_args.lazy_packing:
        data_args.use_packed_ds = True
    training_args.use_packed_ds = data_args.use_packed_ds
    training_args.dataloader_prefetch_factor = data_args.dataloader_prefetch_factor
    data_args.seed = training_args.seed
    data_args.output_dir = training_args.output_dir

    # assertions for custom fields
    if data_args.mmdoc_format not in MMDOC_FORMATS:
        raise ValueError(
            f"You provided --mmdoc-format {data_args.mmdoc_format} "
            f"but only {MMDOC_FORMATS} are avaiable."
        )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, '
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir) 
        and training_args.do_train 
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, 
        add_eos_token=False, 
        trust_remote_code=True, 
        use_fast=model_args.use_fast_tokenizer
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    print("len(tokenizer)", len(tokenizer))
    print("Num New Tokens:", num_new_tokens)

    # Replacing the attention classes if we do sequence packing
    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from internvl.patch import apply_liger_kernel_to_internvit
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    else:
        logger.info('Loading Vision-Encoder...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLM...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        
        # NOTE: up to here, the token_emb matrix and the lm_head are tied 
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)

        # then in this config they manually set the following:
        # - self.tie_word_embeddings = False
        # - self.llm_config.tie_word_embeddings = self.tie_word_embeddings
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), 
            llm_config.to_dict(), 
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, 
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, 
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, 
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, 
            max_dynamic_patch=data_args.max_dynamic_patch
        )
        internvl_chat_config.force_image_size = data_args.force_image_size
        
        # and here they create the model, but since the model is initialized from PretrainedModel 
        # instances, it will just not use the updated config with tie_word_embeddings = False
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    
    model.img_context_token_id = img_context_token_id
    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    if model_args.model_name_or_path is None:
        num_params_llm = sum([params.numel() for params in llm.parameters()])
        num_params_vision = sum([params.numel() for params in vision_model.parameters()])
    num_params_mlp = sum([params.numel() for params in model.mlp1.parameters()])
    num_params_total = sum([params.numel() for params in model.parameters()])

    if model_args.model_name_or_path is None:
        logger.info(f'num_params_llm: {num_params_llm / 1e6}M')
        logger.info(f'num_params_vision: {num_params_vision / 1e6}M')
    logger.info(f'num_params_mlp: {num_params_mlp / 1e6}M')
    logger.info(f'num_params_total: {num_params_total / 1e6}M')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'patch_size of vision-encoder: {patch_size}')
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    logger.info(f'model.num_image_token: {model.num_image_token}')

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        ### TODO: maybe we want to experiment with paligemma init?
        ### paligemma (https://arxiv.org/abs/2407.07726) fig 6 advocates for gaussian noise init
        ### currently, the AvgEmb option is advocated in https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        ### actually, maybe the AvgEmb init is already happening by default, see this PR: https://github.com/huggingface/transformers/pull/33325
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)


    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()


    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        base_collate = lazy_packed_collate_fn if data_args.lazy_packing else packed_collate_fn
        collator = partial(
            base_collate,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

    # resume from checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_dataset = build_datasets(
        data_args, 
        tokenizer, 
        tcs_loader, 
        model, 
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, 
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, 
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type, 
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
        checkpoint=checkpoint
    )

    # build the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[CheckpointCallback(), ThroughputLogCallback()]
    )

    # Auto-calibrate max_steps when a token budget is available but estimated_tokens was not set.
    # This replaces the manual estimate_tokens.sh pre-flight run.
    if (
        training_args.do_train
        and data_args.estimated_tokens == 0
        and data_args.scale is not None
        and data_args.scale in SCALE_CONFIGS
    ):
        scale_cfg = SCALE_CONFIGS[data_args.scale]
        tot_tokens = scale_cfg['tot_tokens']
        batch_size = scale_cfg['batch_size']

        if checkpoint is not None:
            state_path = os.path.join(checkpoint, 'trainer_state.json')
            with open(state_path, 'rb') as f:
                new_max_steps = json.loads(f.read())['max_steps']
            if dist.get_rank() == 0:
                logger.warning(f"[calibration] Resuming from checkpoint; restoring max_steps={new_max_steps} from {state_path}")
        else:
            n_cal = max(1, data_args.calibration_batches // dist.get_world_size())
            if dist.get_rank() == 0:
                logger.warning(f"[calibration] estimated_tokens not set; running {data_args.calibration_batches} batches ({n_cal} steps/rank) to estimate avg tokens per packed sequence...")
            avg_tokens_val = estimate_avg_tokens(trainer.get_train_dataloader(), n_cal)
            new_max_steps = int(tot_tokens / (avg_tokens_val * batch_size))
            if dist.get_rank() == 0:
                logger.warning(
                    f"[calibration] estimated_tokens={avg_tokens_val} "
                    f"=> max_steps={new_max_steps} "
                    f"(tot_tokens={int(tot_tokens)}, batch_size={batch_size})"
                )
        trainer.args.max_steps = new_max_steps

    if training_args.do_train:
        # NOTE: Matteo 13/02, patch to show system metrics for *all* nodes
        # on wandb in multi-node training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_multinode = dist.get_world_size() > torch.cuda.device_count()
        if local_rank == 0 and is_multinode:
            wandb_name = os.environ.get("RUN_NAME")
            group_rank = os.environ.get("GROUP_RANK")
            project = os.environ.get("WANDB_PROJECT", "dcvlm-train")
            team = os.environ.get("WANDB_TEAM") or None
            wandb.init(
                entity=team,
                project=project,
                name=f"group_{group_rank}", 
                group=wandb_name
            )
            wandb.log({"ping":1})

        # ensure the latest checkpoint
        # - is always HF compatible
        # - matches the current model config (in case it has changed)
        # (this is useful for the tie_word_embeddings config param)
        is_master = dist.get_rank() == 0
        if is_master and checkpoint is not None:
            make_transformers_compatible(checkpoint)
            fix_checkpoint_config_mismatch(checkpoint, model)
        
        # train
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        # fix the output of save_model()
        if is_master:
            make_transformers_compatible(training_args.output_dir)

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    
    else:
        avg_fw_tokens = estimate_avg_tokens(trainer.get_train_dataloader(), training_args.max_steps)
        if dist.get_rank() == 0:
            print(f"\n{'='*10} Average tokens per packed sequence = {avg_fw_tokens} {'='*10}")

    
    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == '__main__':
    main()
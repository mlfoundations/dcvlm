# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import os.path as osp
import gc
import copy
import time
import json
import pickle
import logging
from collections import defaultdict, OrderedDict
from typing import List, Union, Dict

import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import get_last_checkpoint
from webdataset.utils import pytorch_worker_info


from .data.pipelinev2 import build_all_pipelines, build_data_mix, StatefulSamples, StatefulUrls
from .data.mix import DeterministicRandomMix
from internvl.mem_utils import format_mem_info

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LazyPackedDataset(IterableDataset):
    def __init__(
        self,
        data_args,
        tokenizer,
        num_images_expected: int = 6,
        max_packed_tokens: int = 32768,
        max_buffer_size: int = 100,
        allow_distributed_replacement: bool = True,
        allow_deduplicated_ds_name: bool = False,
        log_freq: int = 500,
        cleanup_freq: int = 5000,
        seed: int = None,
        checkpoint: str = None,
        collate_dtypes: str = "",
        mmdoc_format: str = "mt"
    ):
        super().__init__()
        
        # params for the wds pipelines
        self.data_args = copy.deepcopy(data_args)
        self.tokenizer = tokenizer
        self.seed = seed

        # packing params
        self.max_buffer_size = max_buffer_size
        self.max_packed_tokens = max_packed_tokens
        self.num_images_expected = num_images_expected

        # misc
        self.allow_distributed_replacement = allow_distributed_replacement
        self.allow_deduplicated_ds_name = allow_deduplicated_ds_name
        self.log_freq = log_freq
        self.cleanup_freq = cleanup_freq
        self.dump_freq = 100
        self.checkpoint = checkpoint

        # list of dtypes to collate
        self.collate_dtypes = collate_dtypes
        self.mmdoc_format = mmdoc_format


    def _check_datasets(self, datasets, probs):
        # avoid duplicates 
        ds_name_list = [d.ds_name for d in datasets]
        if not self.allow_deduplicated_ds_name:
            if len(ds_name_list) != len(set(ds_name_list)):
                raise ValueError(f'Found duplicated ds_name: {ds_name_list}')

    def find_buffer(self, buffer_list, new_sample):
        # search for an existing buffer to host the new sample
        find = False
        find_idx = -1
        num_images_current = new_sample['pixel_values'].size(0)
        for buffer_idx, buffer in enumerate(buffer_list):
            num_images_buffer = buffer['pixel_values'].size(0)    
            # check if images fit
            if num_images_buffer + num_images_current <= self.num_images_expected:
                # check if tokens fit
                num_merged_tokens = new_sample['input_ids'].size(0) + buffer['input_ids'].size(0)
                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

        if find:
            return buffer_list.pop(find_idx)
        return None

    def update_buffer(self, buffer, new_sample):
        # init a new buffer if needed
        if buffer is None:
            new_sample['data_index'] = torch.zeros_like(new_sample['input_ids'])
            return new_sample

        # or concat the sample to an existing buffer
        new_sample['data_index'] = torch.ones_like(new_sample['input_ids']) + buffer['data_index'][-1].item()
        assert buffer.keys() == new_sample.keys()
        for k in buffer:
            buffer[k] = torch.cat([buffer[k], new_sample[k]])
        return buffer
    
    def update_buffer_list(self, buffer_list, buffer):
        # this insertion ensures buffers are always sorted from
        # fullest to emptiest (according to num_images therein)
        find_idx = len(buffer_list)
        num_images_new_sample = buffer['pixel_values'].size(0)
        for buffer_idx in range(len(buffer_list)):
            if buffer_list[buffer_idx]['pixel_values'].size(0) < num_images_new_sample:
                find_idx = buffer_idx
                break
        buffer_list.insert(find_idx, buffer)

        for i in range(1, len(buffer_list)):
            assert buffer_list[i-1]['pixel_values'].size(0) >= buffer_list[i]['pixel_values'].size(0)

        return buffer_list

    def _cleanup(self, buffer_list: List[Dict[str, torch.Tensor]]):
        for buffer in buffer_list:
            for k, v in buffer.items():
                assert v.device.type == "cpu"
                del v
            del buffer
        del buffer_list
        gc.collect()
        torch.cuda.empty_cache()

    def _should_save_state(self):
        # only save state if a new checkpoint is found
        curr_ckpt = get_last_checkpoint(self.data_args.output_dir)
        if curr_ckpt != self.checkpoint:
            self.checkpoint = curr_ckpt
            return True
        return False

    def load_state(self):
        # default init with empty states
        if self.checkpoint is None:
            return defaultdict(str), defaultdict(str), defaultdict(int), [], None

        # otherwise init from file
        rank, world_size, worker_id, num_workers = pytorch_worker_info(group=None)
        state_dir = osp.join(self.checkpoint, 'dataset_state')
        state_file = osp.join(state_dir, f'rank{rank}_worker{worker_id}.json')
        if not osp.exists(state_file):
            raise FileNotFoundError(f'Cannot find dataset state file: {state_file}')
        with open(state_file, 'r') as f:
            state = json.load(f)
        url_states = state['url_states']
        sample_states = state['sample_states']
        epoch_states = state['epoch_states']

        # also load the packing buffers if there
        buffer_file = osp.join(state_dir, f'rank{rank}_worker{worker_id}_buffers.pt')
        if osp.exists(buffer_file):
            buffer_list = torch.load(buffer_file)
        else:
            buffer_list = []

        # load the rng state if there as well
        rng_file = osp.join(state_dir, f'rank{rank}_worker{worker_id}_state.pkl')
        if osp.exists(rng_file):
            with open(rng_file, "rb") as f:
                rng_state = pickle.load(f)
        else:
            rng_state = None

        return url_states, sample_states, epoch_states, buffer_list, rng_state
    
    def set_state(self, url_states, sample_states, epoch_states, rng_state):
        # base case (no state to set bc we are starting fresh)
        if self.checkpoint is None:
            return

        # set the state for all pipelines
        for ds_name, dataset in zip(self.names, self.sources):
            # preliminary checks
            assert isinstance(dataset, wds.DataPipeline)
            
            # effectively set the state
            for step in dataset.pipeline:
                if isinstance(step, StatefulUrls):
                    step.url = url_states[ds_name]["url"]
                    step.last_seen = url_states[ds_name]["last_seen"]
                if isinstance(step, StatefulSamples):
                    step.sample_key = sample_states[ds_name]
                if isinstance(step, wds.detshuffle):
                    # we need to decrement the epoch by 1 here
                    # because __iter__ will increment it
                    # the minimum epoch is -1 (i.e., start from 0)
                    # see https://github.com/webdataset/webdataset/blob/0773837ecd298587fc89c4f944ef346ef1a6b619/src/webdataset/filters.py#L412
                    epoch = max(epoch_states[ds_name]-1, -1)
                    step.epoch = epoch
        
        # set the state for the random-mix 
        for step in self.stream.pipeline:
            if isinstance(step, DeterministicRandomMix):
                step.set_state(rng_state)
                break


    def save_state(self, buffer_list=[]):
        # grab current url, epoch (i.e., repetition), and sample 
        # for *all* pipelines
        url_states, sample_states, epoch_states = {}, {}, {}
        for ds_name, dataset in zip(self.names, self.sources):
            assert isinstance(dataset, wds.DataPipeline)
            for step in dataset.pipeline:
                if isinstance(step, StatefulUrls):
                    if ds_name not in url_states:
                        url_states[ds_name] = {}
                    url_states[ds_name]["url"] = step.url
                    url_states[ds_name]["last_seen"] = step.last_seen
                if isinstance(step, StatefulSamples):
                    sample_states[ds_name] = step.sample_key
                if isinstance(step, wds.detshuffle):
                    epoch_states[ds_name] = step.epoch

        # grab rng state for random-mix
        rng_state = None
        for step in self.stream.pipeline:
            if isinstance(step, DeterministicRandomMix):
                rng_state = step.get_state()
                break

        # create the ckpt dir if needed (only on rank 0 worker 0 to avoid race conds)
        rank, world_size, worker_id, num_workers = pytorch_worker_info(group=None)
        state_dir = osp.join(self.checkpoint, 'dataset_state')
        if rank == 0 and worker_id == 0:
            os.makedirs(state_dir, exist_ok=True)

        # some worker may come before r0, w0, so we sync with a while loop
        # (this is a manual equivalent of dist.barrier())
        while not osp.exists(state_dir):
            time.sleep(0.03)
        
        # Retry params for the three writes below. Exponential backoff handles
        # the Lustre dentry-cache race where state_dir's parent lookup can still
        # ENOENT even after the rank-0 makedirs completes and exists() returns True.
        _MAX_TRIES = 10
        _BASE_DELAY = 0.05  # seconds; doubles each attempt → ~51s total worst case

        # dump the iteration state for individual datasets
        time.sleep(0.05)
        state_file = osp.join(state_dir, f'rank{rank}_worker{worker_id}.json')
        state = {
            'url_states': url_states,
            'sample_states': sample_states,
            'epoch_states': epoch_states
        }
        for attempt in range(_MAX_TRIES):
            try:
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                if attempt > 0:
                    print(f'[dataset_state] rank{rank} worker{worker_id} json save succeeded on attempt {attempt + 1}')
                break
            except Exception as e:
                if attempt == _MAX_TRIES - 1:
                    print(f'[dataset_state] rank{rank} worker{worker_id} json save FAILED after {_MAX_TRIES} attempts: {type(e).__name__}: {e}')
                    raise
                time.sleep(_BASE_DELAY * (2 ** attempt))

        # dump the state of the packing buffers
        if len(buffer_list) > 0:
            buffer_file = osp.join(state_dir, f'rank{rank}_worker{worker_id}_buffers.pt')
            for attempt in range(_MAX_TRIES):
                try:
                    torch.save(buffer_list, buffer_file)
                    if attempt > 0:
                        print(f'[dataset_state] rank{rank} worker{worker_id} buffers save succeeded on attempt {attempt + 1}')
                    break
                except Exception as e:
                    if attempt == _MAX_TRIES - 1:
                        print(f'[dataset_state] rank{rank} worker{worker_id} buffers save FAILED after {_MAX_TRIES} attempts: {type(e).__name__}: {e}')
                        raise
                    time.sleep(_BASE_DELAY * (2 ** attempt))

        # dump the state of the rng for random mix
        if rng_state is not None:
            state_file = osp.join(state_dir, f'rank{rank}_worker{worker_id}_state.pkl')
            for attempt in range(_MAX_TRIES):
                try:
                    with open(state_file, "wb") as f:
                        pickle.dump(rng_state, f)
                    if attempt > 0:
                        print(f'[dataset_state] rank{rank} worker{worker_id} rng save succeeded on attempt {attempt + 1}')
                    break
                except Exception as e:
                    if attempt == _MAX_TRIES - 1:
                        print(f'[dataset_state] rank{rank} worker{worker_id} rng save FAILED after {_MAX_TRIES} attempts: {type(e).__name__}: {e}')
                        raise
                    time.sleep(_BASE_DELAY * (2 ** attempt))


    def __iter__(self):
        # this will loop indefinitely
        self.pid = os.getpid()

        # set up stuff for logging
        rank, world_size, worker_id, num_workers = pytorch_worker_info(group=None)
        logger.error(f"[{rank=}, {worker_id=}] Start yielding.")
        should_log = (rank == 0) and (worker_id == 0)

        # load dataset state if it exists
        # url_states -> last url seen for each dataset
        # sample_states -> last sample key seen within the last url
        # epoch_states -> number of restarts for a single source (not used with replacement)
        # buffer_list -> persisted sequence packing buffers across runs
        # rng_state -> used for deterministic random mixing 
        self.url_states, self.sample_states, self.epoch_states, buffer_list, rng_state = self.load_state()
            
        # initialize the pipelines lazily, only on demand
        self.sources, self.probs, self.names, self.metas = build_all_pipelines(
            self.data_args,
            meta_path=self.data_args.meta_path,
            allow_distributed_replacement=self.allow_distributed_replacement,
            seed=self.seed,
            collate_dtypes=self.collate_dtypes,
            mmdoc_format=self.mmdoc_format
        )

        # this mixes from the various sources and yields
        # after transforming and tokenizing
        self.stream = build_data_mix(
            self.sources,
            self.probs,
            self.names,
            self.metas,
            tokenizer=self.tokenizer,
            template_name=self.data_args.conv_style,
            num_image_token=self.data_args.num_image_token,
            image_size=self.data_args.force_image_size,
            pad2square=self.data_args.pad2square,
            use_thumbnail=self.data_args.use_thumbnail,
            dynamic_image_size=self.data_args.dynamic_image_size,
            min_dynamic_patch=self.data_args.min_dynamic_patch,
            normalize_type=self.data_args.normalize_type,
            max_images=self.num_images_expected,
            max_tokens=self.max_packed_tokens,
            use_packed_ds=True,
            allow_distributed_replacement=self.allow_distributed_replacement,
            seed=self.seed
        )

        # set the state for all pipelines (sets attributes for resuming)
        self.set_state(self.url_states, self.sample_states, self.epoch_states, rng_state)

        # set up stats
        if not hasattr(self, "stats"):
            self.stats = defaultdict(int)

        # inner loop through the data mix
        # - this is always infinite
        for iter_idx, current_sample in enumerate(self.stream):

            # strip useless keys and update stats
            current_sample.pop("__key__")
            ds_name = current_sample.pop("__ds_name__")
            self.stats[ds_name] += 1

            # scan current buffers until one is found that can host the new sample
            # if none exists, the new sample gives rise to a new buffer
            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
        
            # sorted insertion within the buffer, keeps them from fullest to emptiest
            buffer_list = self.update_buffer_list(buffer_list, buffer)

            # make sure nothing is wrong and we don't yield overflown buffers
            # (this should never happen, but for extra security)
            while (
                len(buffer_list) > 0 
                and (
                    buffer_list[0]['pixel_values'].size(0) > self.num_images_expected
                    or buffer_list[0]['input_ids'].size(0) > self.max_packed_tokens
                )
            ):
                if should_log:
                    logger.error(
                        f"Found a buffer with {buffer_list[0]['pixel_values'].size(0)} images "
                        f"and {buffer_list[0]['input_ids'].size(0)} tokens. This is too large."
                    )
                b = buffer_list.pop(0)
                del b

            # yield fully packed buffers
            # fully packed -> we don't have capacity for neither tokens or images
            while (
                len(buffer_list) > 0 
                and (
                    buffer_list[0]['pixel_values'].size(0) == self.num_images_expected
                    or buffer_list[0]['input_ids'].size(0) == self.max_packed_tokens
                )
            ):
                yield buffer_list.pop(0)

            # yield extra buffers if we have too many
            while len(buffer_list) > self.max_buffer_size:
                if should_log:
                    logger.error(
                        f"Failed to pack data to exactly {self.num_images_expected} images, "
                        f"or exactly {self.max_packed_tokens} tokens. "
                        f"Yield a buffer with {buffer_list[0]['pixel_values'].size(0)} images "
                        f"and {buffer_list[0]['input_ids'].size(0)} tokens."
                    )
                yield buffer_list.pop(0)

            # log periodically
            if (iter_idx+1) % self.log_freq == 0:
                # log data distribution
                num_samplings = sum(self.stats.values())
                assert num_samplings == iter_idx + 1
                freqs = {ds_name: v/num_samplings*100 for ds_name, v in self.stats.items()}
                freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
                # log memory usage
                mem_info = format_mem_info(self.pid, tensors=buffer_list)
                logger.error(
                    f"[{rank=}, {worker_id=}] "
                    f"Distribution summary at step {iter_idx+1}:\n"
                    f"{json.dumps({k: f'{v:.2f}%' for k, v in freqs.items()}, indent=2)}\n"
                    f"Mem Info: {mem_info}"
                )

            # periodically throws away suboptimal buffers
            if (iter_idx+1) % self.cleanup_freq == 0:
                logger.error(
                    f"[{rank=}, {worker_id=}] "
                    f"Cleaning up after step {iter_idx+1}."
                )
                self._cleanup(buffer_list)
                buffer_list = []

            # save dataset state if needed
            # this will only get triggered after a new model ckpt is found
            if self._should_save_state():
                self.save_state(buffer_list)

        # we should never get here, so log an error if it happens
        logger.error("Ending the dataset iterator, this should never happen.")


    @staticmethod
    def get_cu_seqlens_and_indexes(
        data_index: torch.LongTensor,  # (seq_len,)
        input_ids: torch.LongTensor,   # (seq_len,)
        labels: torch.LongTensor,   # (seq_len,)
        len2weight: callable,
    ):
        indexes = []
        cu_seqlens = [0]
        loss_weight = []

        start = data_index.min()
        end = data_index.max() + 1
        for i in range(start, end):
            num_tokens = (data_index == i).sum().item()
            indexes.extend(list(range(num_tokens)))
            cu_seqlens.append(cu_seqlens[-1] + num_tokens)
            assert num_tokens > 0

            curr_data_index = data_index[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
            assert (curr_data_index == i).all(), data_index

            curr_labels = labels[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
            # NOTE: this is the number of tokens used for training (actually used for the loss)!!!
            num_effective_tokens = (curr_labels != IGNORE_TOKEN_ID).sum().item()
            loss_weight.extend([len2weight(num_effective_tokens)] * num_tokens)

        assert len(indexes) == data_index.size(0), f'{len(indexes)=}, {data_index.size(0)=}'

        loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
        return cu_seqlens, indexes, loss_weight


WARNING_CNT = defaultdict(int)


def lazy_packed_collate_fn(
    features,
    data_collator,
    len2weight: callable,
    max_item_length: int,
    micro_num: int = 1,
    loss_reduction_all_gather: bool = False,
    pad_id: int = 0,
):
    if not isinstance(features, list):
        features = [features]

    if len(features) > micro_num:
        raise NotImplementedError(f'{len(features)=} > {micro_num=}')

    if len(features) < micro_num and WARNING_CNT['micro_num_warning'] < 5:
        logger.warning(
            f'{len(features)=} > {micro_num=}, '
            f'the features will be padded to satisfy micro_num requirement'
        )
        WARNING_CNT['micro_num_warning'] += 1

    # ensure that the len(features) is equal to the required micro_num
    num_features = len(features)
    while len(features) < micro_num:
        features.append(copy.deepcopy(features[0]))
        features[-1]['labels'] = torch.full_like(features[-1]['labels'], IGNORE_TOKEN_ID)

    indexes = []
    cu_seqlens = []
    cu_num_images_list = [0]

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]  # this will be same shape as close to max_packed_seq_len

    num_samples = 0
    num_padding_tokens = 0
    for feat_idx, feat in enumerate(features):
        data_index = feat.pop('data_index')
        curr_cu_seqlens, curr_indexes, curr_loss_weight = LazyPackedDataset.get_cu_seqlens_and_indexes(
            data_index=data_index,
            input_ids=feat['input_ids'],
            labels=feat['labels'],
            len2weight=len2weight,
        )

        feat['loss_weight'] = curr_loss_weight

        if feat_idx < num_features:
            num_samples += len(curr_cu_seqlens) - 1

        if curr_cu_seqlens[-1] < max_item_length:
            curr_cu_seqlens.append(max_item_length)
            curr_indexes.extend(list(range(max_item_length - curr_cu_seqlens[-2])))

        indexes.append(torch.tensor(curr_indexes, dtype=torch.long))
        cu_seqlens.append(torch.tensor(curr_cu_seqlens, dtype=torch.int32))

        num_padding_tokens += (max_item_length - feat['input_ids'].size(0))
        cu_num_images_list.append(cu_num_images_list[-1] + feat['pixel_values'].size(0))

    batch = data_collator(features=features, max_item_length=max_item_length, pad_id=pad_id)

    # convert it to list in case it is converted into bf16
    batch['loss_weight'] = torch.where(batch['labels'] == IGNORE_TOKEN_ID, 0, batch['loss_weight']).tolist()
    batch['attention_mask'] = torch.stack(cu_seqlens)
    batch['loss_reduction_all_gather'] = loss_reduction_all_gather
    batch['statistics'] = torch.tensor(
        [
            num_samples,
            num_padding_tokens,
            batch['image_flags'].numel() - batch['image_flags'].sum().item(),
        ],
        dtype=torch.long,
    )
    return batch

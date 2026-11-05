import os
import random
import os.path as osp
import imagehash
import math

from argparse import ArgumentParser
from argparse import Namespace

from glob import glob
from tqdm import tqdm
from PIL import Image
import time
import json

import webdataset as wds
from webdataset import TarWriter
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample, group_by_keys

CUMULATIVE_SUM_FILE = "full_token_sum.txt"
CUMULATIVE_SUM_SAMPLES_FILE = "num_saved_samples.txt"

##### copied from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/data.py#L236-239
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
_SHARD_SHUFFLE_SIZE = 1000

def log_and_continue(exn):
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def tarfile_to_samples_throw_wrapped(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys(files, handler=handler)
    return samples

def collate_(batch):    
    return batch

def make_output_folder(args: Namespace):
    assert isinstance(args.input_folder, str) and osp.exists(args.input_folder) and osp.isdir(args.input_folder)
    os.makedirs(args.output_folder, exist_ok=True)
    return args.output_folder

def decode_filter_json(sample):
    for k in sample.keys():
        if 'filters.json' in k:
            sample[k] = json.loads(sample[k])
    return sample

def save_full_token_counts(output_folder, dataloader):

    total_len = 0
    total_samples = 0

    # compute cumulative sum
    for batch in tqdm(dataloader):
        for dec_sample in batch:
            filters_dict = dec_sample.get("filters.json", {})
            total_len += filters_dict["length_tokens"]
            total_samples += 1

    assert total_len > 0

    # save to the file
    with open(os.path.join(output_folder, CUMULATIVE_SUM_FILE), "w") as f:
        f.write(str(total_len))
    
    return total_len

def write_sample(writer, sample):
    writer.write(sample)

def get_token_limit(pool, size, num_tars):

    pool_multiplier = {
        "small": 1/32.,
        "medium": 1/8.,
        "large": 1/2.,
        "xlarge": 1.,
    }

    return math.ceil( pool_multiplier[pool] * size ), math.ceil( pool_multiplier[pool] * num_tars)

def test(args, sample_count, shard_count, len_to_match):

    # flat list of tarfiles in the output directory
    tarfiles = sorted(glob(osp.join(args.output_folder, "*.tar"), recursive=False))

    assert len(tarfiles) == shard_count, "Number of tars mismatch"

    # get brace-expanded format for wds
    brace_expanded = osp.join(args.output_folder, '{' + '{}..{}'.format(osp.basename(tarfiles[0]).removesuffix(".tar"), osp.basename(tarfiles[-1]).removesuffix(".tar")) + '}.tar')

    # get wds pipeline
    decoded_dataset = wds.DataPipeline(
        wds.SimpleShardList(brace_expanded),  # this will give an iterator over all tars
        wds.split_by_worker,  # this will distribute across all cpu workers within a rank (num_workers per gpu)
        tarfile_to_samples_throw_wrapped,  # this will give us sample dicts per worker
        wds.map(decode_filter_json, handler=log_and_continue),
        wds.batched(args.batch_size, collation_fn=collate_),
    )

    # create dataloader for batched dataloading
    dataloader = wds.WebLoader(
        decoded_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    total_len = 0
    total_samples = 0

    # compute total sum
    for batch in tqdm(dataloader):
        for dec_sample in batch:
            filters_dict = dec_sample.get("filters.json", {})
            total_len += filters_dict["length_tokens"]
            total_samples += 1

    assert total_len > 0

    assert total_len >= len_to_match, "Number of tokens mismatch"
    assert total_samples == sample_count, "Number of samples mismatch"

def main(args):

    # flat list of tarfiles in the current directory
    tarfiles = sorted(glob(osp.join(args.input_folder, "*.tar"), recursive=False))

    # get brace-expanded format for wds
    brace_expanded = osp.join(args.input_folder, '{' + '{}..{}'.format(osp.basename(tarfiles[0]).removesuffix(".tar"), osp.basename(tarfiles[-1]).removesuffix(".tar")) + '}.tar')

    # parent folder to dump new tarfiles
    output_folder = make_output_folder(args)

    # get wds pipeline
    decoded_dataset = wds.DataPipeline(
        wds.SimpleShardList(brace_expanded),  # this will give an iterator over all tars
        wds.split_by_worker,  # this will distribute across all cpu workers within a rank (num_workers per gpu)
        tarfile_to_samples_throw_wrapped,  # this will give us sample dicts per worker
        wds.map(decode_filter_json, handler=log_and_continue),
        wds.batched(args.batch_size, collation_fn=collate_),
    )

    # create dataloader for batched dataloading
    dataloader = wds.WebLoader(
        decoded_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # get time stats
    st = time.time()

    # check if full sum of token lengths exists
    if os.path.exists(input_folder) and os.path.exists(os.path.join(input_folder, CUMULATIVE_SUM_FILE)):
        with open(os.path.join(output_folder, CUMULATIVE_SUM_FILE)) as f:
            total_len = int(f.read())
    else:
        total_len = save_full_token_counts(args.output_folder, dataloader)
    print("====" * 5)
    print("Total number of tokens in folder: {}".format(total_len))
    print("====" * 5)

    # get num tars in input folder
    num_tars = len(os.listdir(args.input_folder))

    # get token limit for current pool
    tok_limit, num_new_tars = get_token_limit(args.pool_size, total_len, num_tars)
    print("====" * 5)
    print(f"Total shards to create: {num_new_tars:,}.")
    print("Total number of tokens to subsample: {}".format(tok_limit))
    print("====" * 5)

    # create the pool
    sink_pattern = os.path.join(args.output_folder, f"%06d.tar")

    # Track samples and shards
    sample_count = 0
    shard_count = 0

    # Track cumulative tokens seen
    cumulative_toks_seen = 0

    # Start reading and writing
    flag = 0
    print("Processing {} tokens to write to {}".format(tok_limit, args.output_folder))
    with wds.ShardWriter(sink_pattern, maxcount=args.samples_per_shard, maxsize=10e10) as writer:
        for index, batch in enumerate(tqdm(dataloader)):

            for dec_sample in batch:

                # compute cumulative tokens
                cumulative_toks_seen += dec_sample["filters.json"]["length_tokens"]

                # persist the original tar that this sample came from in the original full pool
                dec_sample["__original_url__"] = dec_sample["__url__"]
                dec_sample.pop("__url__")

                # write sample
                write_sample(writer, dec_sample)
                sample_count += 1

                # logging
                if sample_count % args.samples_per_shard == 0:
                    print(f"Shard {shard_count} completed with {args.samples_per_shard} samples")
                    shard_count += 1

                # break when done with required tokens
                if cumulative_toks_seen >= tok_limit:
                    flag = 1
                    break

            if flag:
                break

    print(f"Total samples processed: {sample_count} into {shard_count+1} shards!")
    print(f"Full resharded dataset available at: {args.output_folder}")
    print("Took {} seconds to write everything!".format(time.time()-st))

    with open(os.path.join(output_folder, CUMULATIVE_SUM_SAMPLES_FILE), "w") as f:
        f.write(str(sample_count))

    return sample_count, shard_count + 1, tok_limit

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--pool_size', type=str, required=True, choices=["small", "medium", "large", "xlarge"])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument("--samples_per_shard", type=int, default=10_000)
    args = parser.parse_args()
    
    sample_count, shard_count, tok_limit = main(args)

    test(args, sample_count, shard_count, tok_limit)
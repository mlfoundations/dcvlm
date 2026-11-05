import os
import io
import time
import random
import torch
import os.path as osp
import warnings

from argparse import ArgumentParser
from argparse import Namespace

from functools import partial
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageFile, UnidentifiedImageError, PngImagePlugin
from torchvision import transforms
import imagehash

import webdataset as wds
from webdataset import TarWriter
from webdataset.tariterators import url_opener, tar_file_expander, group_by_keys

from utils.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                       IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                       IMG_START_TOKEN, QUAD_END_TOKEN,
                       QUAD_START_TOKEN, REF_END_TOKEN,
                       REF_START_TOKEN)
from utils.dataloading import (build_transform, dynamic_preprocess, preprocess_internvl2_5)
from utils.web_dataloading import LengthFilterWebDataset
from transformers import AutoTokenizer

# some flags to avoid errors
Image.MAX_IMAGE_PIXELS = 30_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

def log_and_continue(exn):
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def tarfile_to_samples_throw_wrapped(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys(files, handler=handler)
    return samples

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

def test(input_folder: str, output_folder: str, args: Namespace, filter_names: list[str], delete_old: bool, start_tar_index: str, end_tar_index: str):
    print("Running tests (may take a while...)")

    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)
    random.shuffle(tarfiles)

    num_to_test = min(3, int(end_tar_index) - int(start_tar_index) + 1)

    ### NOTE (Vishaal, 15/05) run the test on a five random tars
    if start_tar_index is None:
        selected_input_tarfiles = [osp.join(input_folder, osp.basename(x)) for x in tarfiles[:num_to_test]]  # pick any 5 at random
        selected_output_tarfiles = [osp.join(output_folder, osp.basename(x)) for x in tarfiles[:num_to_test]]  # pick any 5 at random
    else:    
        # Determine padding width from the input
        padding_width = len(start_tar_index)
        start_index_int = int(start_tar_index)

        # Generate filenames: start_tar_index, start_tar_index+1, ..., +4
        selected_input_tarfiles = []
        selected_output_tarfiles = []
        for i in range(num_to_test):
            index_str = str(start_index_int + i).zfill(padding_width)
            tar_name = f"{index_str}.tar"
            tar_path = osp.join(input_folder, tar_name)
            if osp.exists(tar_path):
                tar_path_out = osp.join(output_folder, tar_name)
                assert osp.exists(tar_path_out), f"Input tar exists {tar_path}, but output tar does not {tar_path_out}"
                assert osp.isfile(tar_path_out), f"{tar_path_out} does not seem to be a valid tarfile. Please manually check this."
                selected_input_tarfiles.append(tar_path)
                selected_output_tarfiles.append(tar_path_out)

    for input_tarfile, output_tarfile in tqdm(zip(selected_input_tarfiles, selected_output_tarfiles), desc=f"Running tests on {len(selected_input_tarfiles)} tars!", ascii=True, total=len(selected_input_tarfiles)):

        # create corresponding datasets and iterators
        input_dataset = wds.WebDataset(input_tarfile).decode("pil", handler=log_and_continue)
        output_dataset = wds.WebDataset(output_tarfile).decode("pil", handler=log_and_continue)
        input_iter = iter(input_dataset)

        ### NOTE (Vishaal, 15/05): we iterate over the output dataset and then seek the corresponding input samples since 
        ### the output dataset is guaranteed to have less than or equal samples than the input, hence we should ALWAYS be
        ### able to seek to the correct output key in the input dataset.
        for out_sample in tqdm(output_dataset):
            # some samples in the original wds may be corrupted / non-decodable
            # those will not have a match in the new output shards
            sample_key = out_sample['__key__']
            inp_sample = seek_key(input_iter, key=sample_key)

            # check filters are there
            assert "filters.json" in out_sample, \
                f"There is no filters.json key in the sample {sample_key}."        
            
            # check the new keys are there
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

    if delete_old and start_tar_index is not None and end_tar_index is not None:
        print("Deleting all tars from start_tar_index to end_tar_index in input folder...")
        padding_width = len(start_tar_index)
        for i in range(int(start_tar_index), int(end_tar_index) + 1):
            tar_name = f"{str(i).zfill(padding_width)}.tar"
            tar_path = osp.join(input_folder, tar_name)
            if osp.exists(tar_path):
                try:
                    if osp.islink(tar_path):
                        real_path = os.readlink(tar_path)
                        real_path_full = osp.join(osp.dirname(tar_path), real_path) \
                            if not osp.isabs(real_path) else real_path

                        # Delete the original file the symlink points to
                        if osp.exists(real_path_full):
                            os.remove(real_path_full)
                            print(f"Deleted target of symlink: {real_path_full}")
                        else:
                            print(f"Symlink target not found: {real_path_full}")

                        # Delete the symlink itself
                        os.remove(tar_path)
                        print(f"Deleted symlink: {tar_path}")
                    else:
                        os.remove(tar_path)
                        print(f"Deleted: {tar_path}")
                except Exception as e:
                    print(f"Failed to delete {tar_path}: {e}")

def main(args):
    # seed libraries that we're using
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)
    # sort them based on the integer key from the filename
    tarfiles = sorted(
        tarfiles,
        key=lambda x: int(osp.basename(x).split(".")[0])
    )

    # get brace-expanded format for wds
    if args.start_tar_index is None or args.end_tar_index is None:
        brace_expanded = osp.join(args.input_folder, '{' + '{}..{}'.format(osp.basename(tarfiles[0]).removesuffix(".tar"), osp.basename(tarfiles[-1]).removesuffix(".tar")) + '}.tar')
    else:
        assert os.path.exists(osp.join(args.input_folder, '{}.tar'.format(args.start_tar_index))), "start tar: {}.tar does not exist in {}!".format(args.start_tar_index, args.input_folder)
        assert os.path.exists(osp.join(args.input_folder, '{}.tar'.format(args.end_tar_index))), "end tar: {}.tar does not exist in {}!".format(args.end_tar_index, args.input_folder)
        brace_expanded = osp.join(args.input_folder, '{' + '{}..{}'.format(args.start_tar_index, args.end_tar_index) + '}.tar')

    # parent folder to dump new tarfiles
    output_folder = make_output_folder(args)

    # get time stats
    st = time.time()

    # get wds pipeline
    decoded_dataset = LengthFilterWebDataset(
        template_name="internvl2_5",  # we always preprocess with the internvl2_5 formatter
        meta={'root': brace_expanded, 'data_type': args.data_type},  # we need to specify where to load all tars from and what is the data-type of current dataset
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_name),  # tokenizer to use for getting token stats
        ds_name=args.ds_name,
        num_image_token=256,  # we hardcode this to 256 since for a fixed image size (448) and downsampling ratio (0.5) we always get 256 tokens!
        image_size=448,  # default image size of InternViT
        is_train=False,  # we do not want to use augmentations!
        pad2square=False,  # we do not use padding!
        dynamic_image_size=True,  # for computing token stats, we need to ensure that we use stats based on dynamic image preprocessing
        use_thumbnail=True,  # again, for token stats to be faithful to the actual usage during training, we need to include the thumbnail!
        batch_size=args.batch_size,
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

    # init writers for each tar
    writers, iterators = {}, {}

    # track full stats
    total_chars = 0
    total_tokens = 0
    total_tokens_text_only = 0
    total_tokens_trainable_only = 0
    total_images = 0
    total_samples = 0

    # sequentially iter through tars 
    for batch in tqdm(dataloader):
        for index, dec_sample in enumerate(batch):
            # we first get the local-path of the current sample
            lp_ = dec_sample["__url__"]

            # get the iterator for this tar if not already loaded
            if lp_ not in iterators:
                try:
                    # we get the iterator
                    iterators[lp_] = iter(wds.WebDataset(lp_, shardshuffle=False))
                except Exception as e:
                    print(f"[warning] Skipping corrupt tar file during write: {lp_} -> {e}")
                    continue    

            # get the encoded sample
            try:
                enc_sample = seek_key(iterators[lp_], key=dec_sample["__key__"])
            except Exception as e:
                print(f"[warning] Failed to seek key in tar {lp_}: {e}")
                continue

            # add to current sample
            enc_sample["filters.json"] = dec_sample["filters.json"]

            # get tar writer
            if lp_ not in writers:
                # get path to written tar
                ot_ = osp.join(output_folder, osp.basename(lp_))
                # we create writer
                writers[lp_] = TarWriter(ot_)

            # write the sample
            writers[lp_].write(enc_sample)

            # track stats
            total_samples += 1
            total_chars += enc_sample["filters.json"][decoded_dataset.filter_name_char]
            total_tokens += enc_sample["filters.json"][decoded_dataset.filter_name_tokens]
            total_tokens_text_only += enc_sample["filters.json"][decoded_dataset.filter_name_tokens_text_only]
            total_tokens_trainable_only += enc_sample["filters.json"][decoded_dataset.filter_name_tokens_trainable_only]
            total_images += enc_sample["filters.json"][decoded_dataset.filter_name_images]

    # close all writer streams
    for lp_ in writers:
        writers[lp_].close()    

    print(f"Total samples: {total_samples}, Total characters: {total_chars}, Total tokens: {total_tokens}, Total text-only tokens: {total_tokens_text_only}, Total trainable-only tokens: {total_tokens_trainable_only}, Total images: {total_images}")
    print("Took {:.2f} seconds to write everything!".format(time.time()-st))

    # return paths for testing
    return args.input_folder, output_folder, decoded_dataset.filter_names


if __name__ == "__main__":
    # very simple cmd-line args, feel free to extend
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--filter_name', type=str, default='length')
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--start_tar_index', type=str, required=False, default=None, 
                        help="If set, the start index in full string format (eg, '00000') to add in the brace-expanded string of webdataset, else will automatically construct the brace-expanded string")
    parser.add_argument('--end_tar_index', type=str, required=False, default=None, 
                        help="If set, the end index in full string format (eg, '00100') to add in the brace-expanded string of webdataset, else will automatically construct the brace-expanded string")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--data_type', type=str, default=None, required=True)
    parser.add_argument('--tokenizer_name', type=str, default='OpenGVLab/InternVL2_5-1B')
    parser.add_argument('--delete_old', action="store_true", help="whether to delete the input_files and only keep the output_files")
    args = parser.parse_args()
    
    # do your filtering here
    input_folder, output_folder, filter_names = main(args)

    assert args.start_tar_index is not None, "this script HAS to use a start_tar_index"
    assert args.end_tar_index is not None, "this script HAS to use a end_tar_index"

    # and run a simple unit-test
    # NOTE: this may not cover all cases, especially sample-wise key checks
    # but we can take it as a reference / starting point
    test(input_folder, output_folder, args=args, filter_names=filter_names, delete_old=args.delete_old, start_tar_index=args.start_tar_index, end_tar_index=args.end_tar_index)
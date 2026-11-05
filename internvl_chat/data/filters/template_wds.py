import os
import random
import os.path as osp
import imagehash

from argparse import ArgumentParser
from argparse import Namespace

from glob import glob
from tqdm import tqdm
from PIL import Image
import time

import webdataset as wds
from webdataset import TarWriter


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


def test(input_folder: str, output_folder: str, filter_name: str):
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


def main(args):

    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)

    # parent folder to dump new tarfiles
    output_folder = make_output_folder(args)

    # get time stats
    st = time.time()

    # sequentially iter through tars (we may consider multiprocessing here)
    for tarfile in tarfiles:

        # init a WebDataset with the current tar
        encoded_dataset = wds.WebDataset(tarfile)
        decoded_dataset = wds.WebDataset(tarfile).decode("pil", handler=log_and_continue)
        encoded_iter = iter(encoded_dataset)
        
        # programmatically create output tar
        output_tar = osp.join(output_folder, osp.basename(tarfile))
        print(f"Will write to {output_tar}")
        with TarWriter(output_tar) as writer:
            
            # iterate through the samples in the decoded WebDataset 
            for dec_sample in decoded_dataset:

                # seek the corresponding encoded sample
                enc_sample = seek_key(encoded_iter, key=dec_sample['__key__'])
                
                # TODO: implement stuff here!
                # (this example is a silly filter with random scoring yet again)
                filters_dict = dec_sample.get("filters.json", {})
                filters_dict[args.filter_name] = float(random.randint(1, 10))
                
                # write the encoded sample on disk (storage-friendly)
                enc_sample["filters.json"] = filters_dict
                writer.write(enc_sample)
    
    print("Took {} seconds to write everything!".format(time.time()-st))

    # return paths for testing
    return args.input_folder, output_folder



if __name__ == "__main__":
    # very simple cmd-line args, feel free to extend
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--filter_name', type=str, required=True)
    args = parser.parse_args()
    
    # do your filtering here
    input_folder, output_folder = main(args)

    # and run a simple unit-test
    # NOTE: this may not cover all cases, especially sample-wise key checks
    # but we can take it as a reference / starting point
    test(input_folder, output_folder, args.filter_name)


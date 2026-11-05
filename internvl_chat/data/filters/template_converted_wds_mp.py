import os
import random
import os.path as osp
import imagehash
import multiprocessing as mp

from argparse import ArgumentParser
from argparse import Namespace

from functools import partial

from glob import glob
from tqdm import tqdm
from PIL import Image

import webdataset as wds
from webdataset import TarWriter
import time
import re
from PIL import Image, ImageFile, UnidentifiedImageError, PngImagePlugin

# some flags to avoid errors
Image.MAX_IMAGE_PIXELS = 30_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

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


# Helper for processing the conversations.txt file and classifying each sample according to DMT taxonomy
# input   : decoded_sample from the WDS tar
# output  : a tuple of (human phrases, gpt phrases, image count and location, sample's class in the taxonomy)
def get_conversations_and_sample_classification(sample):
    conversations = sample.get("conversations.txt", {})
    human_phrases = []
    gpt_phrases = []
    image_info = []

    is_grounding = False
	
    # Locate and separate conversations based on <EOCL>
    turns = re.findall(r'(human:.*?(?:<EOCL>|$))|(gpt:.*?(?:<EOCL>|$))', conversations, re.DOTALL)
    for human_turn, gpt_turn in turns:

        if human_turn:

            # Clean the conversation from the <EOCL> and <image> tags before appending
            text = human_turn[len("human:"):].strip()
            text = re.sub(r'<EOCL>$', '', text).strip()
            text = re.sub(r'Image\d+:\s*(<image>)', r'\1', text)

            # Having opening and closing <box> tags only occurs for the "Grounding" samples
            if "<box>" in text and "</box>" in text:
               is_grounding = True

            human_phrases.append(text)
			
            # Extract image information, specifically:
            # count    : number of <image> tags in the human prompt
			# position : whether <image> comes before or body of the prompt 
            image_count = text.count("<image>")
            if image_count > 0:
                image_positions = [m.start() for m in re.finditer(r"<image>", text)]
                first_image_pos = image_positions[0]
                first_text_pos = len(text)

                for m in re.finditer(r"[^\s<>]", text):
                    if m.group(0) != "<" and m.start() != first_image_pos:
                        first_text_pos = m.start()
                        break
                position = "before_text" if first_image_pos < first_text_pos else "after_text"
                image_info.append({
                    "count": image_count,
                    "position": position,
                })

        elif gpt_turn:

            # Clean the conversation from the <EOCL> tags before appending
            text = gpt_turn[len("gpt:"):].strip()
            text = re.sub(r'<EOCL>$', '', text).strip()
            gpt_phrases.append(text)

            # Having opening and closing <box> tags only occurs for the "Grounding" samples
            if "<box>" in text and "</box>" in text:
               is_grounding = True

    total_image_count = sum([prompt_image['count'] for prompt_image in image_info])
    
    # Multi-turn is set to true only if there are at least two prompts and two responses 
    is_multi_turn_text = len(gpt_phrases) > 1 and len(human_phrases) > 1

    # InternVL ChatFormat classification for the primary class of the taxonomy
    if total_image_count == 0:
        internvl_format = "pure-text"
    elif is_grounding:
        internvl_format = "grounding"
    elif total_image_count > 1:
        internvl_format = "multi-image"
    else:
        internvl_format = "single-image"
	
    sample_cls = {"internvl_format": internvl_format, "is_multi_turn_text": is_multi_turn_text}

    return human_phrases, gpt_phrases, image_info, sample_cls

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


def score_and_dump_tarfile(tarfile: str, output_folder: str):
    
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
            
		    # Below is a sample call for parsing the conversations and classifying the sample
            # based on the taxonomy in the DMT - Dataset Taxonomy sheet.
            human_phrases, gpt_phrases, image_info, sample_cls = get_conversations_and_sample_classification(dec_sample)
            
			# seek the corresponding encoded sample
            enc_sample = seek_key(encoded_iter, key=dec_sample['__key__'])
            
            # TODO: implement stuff here!
            # (this example is a silly filter with random scoring yet again)
            filters_dict = dec_sample.get("filters.json", {})
            filters_dict[args.filter_name] = float(random.randint(1, 10))
            
            # write the encoded sample on disk (storage-friendly)
            enc_sample["filters.json"] = filters_dict
            writer.write(enc_sample)

    return output_tar


def main(args):

    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)

    # parent folder to dump new tarfiles
    output_folder = make_output_folder(args)

    # get time stats
    st = time.time()

    # dump all new tarfiles with injected scores (multiprocessing version)
    # NOTE: the scoring and dumping logic will be in the `score_and_dump_tarfile` function now
    func_for_multiprocessing = partial(score_and_dump_tarfile, output_folder=output_folder)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        
        # imap_unordered is preferred over imap and map
        output_tarfiles = pool.imap_unordered(
            func=func_for_multiprocessing, iterable=tarfiles
        )

        # quick check that things went smoothly
        for idx, output_tarfile in enumerate(output_tarfiles): 
            assert osp.exists(output_tarfile) and osp.isfile(output_tarfile)
            print(f"[{idx+1} / {len(tarfiles)}]\t {output_tarfile} dumped!")
    
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

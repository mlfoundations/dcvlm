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
import re
import shutil

import webdataset as wds
from webdataset import TarWriter
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample, group_by_keys
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
    output_folder = osp.join(args.input_folder, "tmp_" + args.filter_name)
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
  
    taxonomy_dict = {"human_phrases": human_phrases, "gpt_phrases": gpt_phrases, "image_info": image_info, "sample_cls": sample_cls}
    sample["taxonomy_dict"] = taxonomy_dict
    return sample


def test(input_folder: str, output_folder: str, filter_name: str, start_tar_index: str, end_tar_index: str, delete_old: bool):
    print("Running tests (may take a while...)")
    
    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)
    random.shuffle(tarfiles)
	
    # User may also not set the start and end indices. Set both indices 0 in that case
    start_idx = 0 if start_tar_index is None else start_tar_index
    end_idx = 0 if end_tar_index is None else end_tar_index
    
    num_to_test = min(3, int(end_idx) - int(start_idx) + 1)
    
    if start_tar_index is None:
        selected_input_tarfiles = [osp.join(input_folder, osp.basename(x)) for x in tarfiles[:num_to_test]]  # pick any **one** at random
        selected_output_tarfiles = [osp.join(output_folder, osp.basename(x)) for x in tarfiles[:num_to_test]]  # pick any **one** at random

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

        ### NOTE from original WDS template (Vishaal, 15/05): we iterate over the output dataset and then seek the corresponding
        ### input samples since the output dataset is guaranteed to have less than or equal samples than the input, hence we should
        ### ALWAYS be able to seek to the correct output key in the input dataset.
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
            for filter_name in filters:
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

        # Create the metadata folder if it is missing before deleting the old tars
        metadata_folder = osp.join(input_folder, "metadata")
        os.makedirs(metadata_folder, exist_ok=True)
     
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
                        print(f"Deleted old symlink: {tar_path}")

                    else:
                        os.remove(tar_path)
                        print(f"Deleted old tar in: {tar_path}")
                    
                    # Move the updated tar files from the temp folder to the original data folder
                    new_tar_file = osp.join(output_folder, tar_name)
                    dest_new_tar_file = shutil.move(new_tar_file, input_folder) 
                    print(f"Moved edited tar {tar_name} to {input_folder}") 
                    
                    # Save the tar as a success after moving its updated version to the original data folder
                    with open(osp.join(metadata_folder, filter_name + "_successful.txt"), "a+") as f:
                        f.write(tar_name + " successfully added...\n")

                except Exception as e:
                    print(f"Failed to delete {tar_path}: {e}")  

        # Remove the temporary output folder emptying it
        shutil.rmtree(output_folder)
        print(f"Removed the temporary filter folder : {output_folder}")
        
      
def main(args):

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
    decoded_dataset = wds.DataPipeline(
        wds.SimpleShardList(brace_expanded),  # this will give an iterator over all tars
        wds.split_by_worker,  # this will distribute across all cpu workers within a rank (num_workers per gpu)
        tarfile_to_samples_throw_wrapped,  # this will give us sample dicts per worker
        wds.decode("pilrgb", handler=log_and_continue),  # this decodes the images and json
        wds.map(get_conversations_and_sample_classification),
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

    # init writers for each tar
    writers, iterators = {}, {}

    # sequentially iter through tars (we may consider multiprocessing here)
    for batch in tqdm(dataloader):

        # TODO: implement stuff here!
        # (this example is a silly filter with random scoring yet again)
        # note that this can be batched op for faster processing
        fds = []
        for dec_sample in batch:

		    # Below is a sample call for parsing the conversations and classifying the sample
            # based on the taxonomy in the DMT - Dataset Taxonomy sheet.

            filters_dict = dec_sample.get("filters.json", {})
            filters_dict[args.filter_name] = float(random.randint(1, 10))
            fds.append(filters_dict)

        # once we have the processed vals, we do 
        # sequential writing of tars
        for index, dec_sample in enumerate(batch):

            # we first get the local-path of the current sample
            lp_ = dec_sample["__url__"]

            # get the iterator for this tar if not already loaded
            if lp_ not in iterators:
                # we get the iterator
                iterators[lp_] = iter(wds.WebDataset(lp_, shardshuffle=False))

            # get the encoded sample
            enc_sample = seek_key(iterators[lp_], key=dec_sample["__key__"])

            # add to current encoded sample
            enc_sample["filters.json"] = fds[index]

            # get tar writer
            if lp_ not in writers:
                # get path to written tar
                ot_ = osp.join(output_folder, osp.basename(lp_))
                # we create writer
                writers[lp_] = TarWriter(ot_)

            # write the sample
            writers[lp_].write(enc_sample)

    # close all writer streams
    for lp_ in writers:
        writers[lp_].close()    

    print("Took {} seconds to write everything!".format(time.time()-st))

    # return paths for testing
    return args.input_folder, output_folder



if __name__ == "__main__":
    # very simple cmd-line args, feel free to extend
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--filter_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--start_tar_index', type=str, required=False, default=None,
                        help="If set, the start index in full string format (eg, '00000') to add in the brace-expanded string of webdataset, else will automatically construct the brace-expanded string")
    parser.add_argument('--end_tar_index', type=str, required=False, default=None,
                        help="If set, the end index in full string format (eg, '00100') to add in the brace-expanded string of webdataset, else will automatically construct the brace-expanded string")
    parser.add_argument('--delete_old', action="store_true", help="whether to delete the input_files and only keep the output_files")
    args = parser.parse_args()
    
    # do your filtering here
    input_folder, output_folder = main(args)

    # and run a simple unit-test
    # NOTE: this may not cover all cases, especially sample-wise key checks
    # but we can take it as a reference / starting point
    test(input_folder, output_folder, args.filter_name, start_tar_index=args.start_tar_index, end_tar_index=args.end_tar_index, delete_old=args.delete_old)

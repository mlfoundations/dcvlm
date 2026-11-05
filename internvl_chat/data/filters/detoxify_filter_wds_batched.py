import os
import random
import os.path as osp
import imagehash
from concurrent.futures import ThreadPoolExecutor

from argparse import ArgumentParser
from argparse import Namespace

from glob import glob
from tqdm import tqdm
from PIL import Image
import time

import webdataset as wds
from webdataset import TarWriter
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample, group_by_keys

import torch
from torch.amp import autocast
import transformers

### Copied a bunch of code from: https://github.com/unitaryai/detoxify/blob/master/detoxify/detoxify.py
DOWNLOAD_URL = "https://github.com/unitaryai/detoxify/releases/download/"
MODEL_URLS = {
    "original": DOWNLOAD_URL + "v0.1-alpha/toxic_original-c1212f89.ckpt",
    "unbiased": DOWNLOAD_URL + "v0.3-alpha/toxic_debiased-c7548aa0.ckpt",
    "multilingual": DOWNLOAD_URL + "v0.4-alpha/multilingual_debiased-0b549669.ckpt",
    "original-small": DOWNLOAD_URL + "v0.1.2/original-albert-0e1d6498.ckpt",
    "unbiased-small": DOWNLOAD_URL + "v0.1.2/unbiased-albert-c8519128.ckpt",
}
PRETRAINED_MODEL = None
MAX_CHUNKS = 10

def get_model_and_tokenizer(
    model_type, model_name, tokenizer_name, num_classes, state_dict, huggingface_config_path=None
):
    model_class = getattr(transformers, model_name)
    config = model_class.config_class.from_pretrained(model_type, num_labels=num_classes)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=huggingface_config_path or config,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,
        # TODO: may be needed to let it work with Kaggle competition
        # model_max_length=512,
    )

    return model, tokenizer


def load_checkpoint(model_type="original", checkpoint=None, device="cpu", huggingface_config_path=None):
    if checkpoint is None:
        checkpoint_path = MODEL_URLS[model_type]
        loaded = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=device)
    else:
        loaded = torch.load(checkpoint, map_location=device)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    # standardise class names between models
    change_names = {
        "toxic": "toxicity",
        "identity_hate": "identity_attack",
        "severe_toxic": "severe_toxicity",
    }
    class_names = [change_names.get(cl, cl) for cl in class_names]
    model, tokenizer = get_model_and_tokenizer(
        **loaded["config"]["arch"]["args"],
        state_dict=loaded["state_dict"],
        huggingface_config_path=huggingface_config_path,
    )

    return model, tokenizer, class_names


def load_model(model_type, checkpoint=None):
    if checkpoint is None:
        model, _, _ = load_checkpoint(model_type=model_type)
    else:
        model, _, _ = load_checkpoint(checkpoint=checkpoint)
    return model


class Detoxify:
    """Detoxify
    Easily predict if a comment or list of comments is toxic.
    Can initialize 5 different model types from model type or checkpoint path:
        - original:
            model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
        - unbiased:
            model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
        - multilingual:
            model trained on data from the Jigsaw Multilingual
            Toxic Comment Classification Challenge
        - original-small:
            lightweight version of the original model
        - unbiased-small:
            lightweight version of the unbiased model
    Args:
        model_type(str): model type to be loaded, can be either original,
                         unbiased or multilingual
        checkpoint(str): checkpoint path, defaults to None
        device(str or torch.device): accepts any torch.device input or
                                     torch.device object, defaults to cpu
        huggingface_config_path: path to HF config and tokenizer files needed for offline model loading
    Returns:
        results(dict): dictionary of output scores for each class
    """

    def __init__(self, model_type="original", checkpoint=PRETRAINED_MODEL, device="cpu", huggingface_config_path=None):
        super().__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            huggingface_config_path=huggingface_config_path,
        )
        self.class_indices = list(range(len(self.class_names)))
        self.device = device
        self.model.to(self.device)

        if torch.__version__ >= "2.0" and args.device == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("Using compiled model!")
            except Exception as e:
                print(f"[warning] torch.compile failed: {e}, using non-compiled model!")

    #### 14/04 (Vishaal): We patch this function to ensure that 
    #### we can take chunks of 512 size so that we never have
    #### to do truncation
    #### note this doesn't do batching --- we never use it!!
    @torch.no_grad()
    def predict(self, text):

        self.model.eval()

        # get all input-ids for full sequence without truncation
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False, padding=True).to(self.model.device)

        # get the input-ids and attention mask
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # we will now start chunking in 512-size since that's the context len of the roberta model we are using
        chunk_size = 512
        scores_list = []

        # iterate over full sequence until we see all tokens in a chunk-size of 512
        for i in range(0, len(input_ids), chunk_size):
            chunk_input_ids = input_ids[i : i+chunk_size]
            chunk_attention_mask = attention_mask[i : i+chunk_size]

            # pad if chunk is less than chunk_size
            pad_len = chunk_size - chunk_input_ids.shape[0]
            if pad_len > 0:
                chunk_input_ids = torch.nn.functional.pad(chunk_input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
                chunk_attention_mask = torch.nn.functional.pad(chunk_attention_mask, (0, pad_len), value=0)

            # get the chunk to process
            chunk = {
                "input_ids": chunk_input_ids.unsqueeze(0).to(self.model.device),
                "attention_mask": chunk_attention_mask.unsqueeze(0).to(self.model.device),
            }

            # get current chunk scores
            with autocast(device_type='cuda', enabled=(self.device == "cuda")):
                out = self.model(**chunk)[0]
                scores = torch.sigmoid(out).cpu()
            scores_list.append(scores)

        # Aggregate scores across chunks with max since we only care about the chunk with the maximum scores
        # Note: this might introduce some weird artefacts when splitting up some text sequences but its a
        # a reasonable approximation to what we need!
        all_scores = torch.stack(scores_list).max(dim=0).values

        results = []
        # If the input is a single text, squeezing will remove the dimensionality from the tensor -
        # so `.tolist()` will return a number instead. Otherwise, we'll get the list of scores of that class.
        for i, cla in enumerate(self.class_names):
            results.append(all_scores[:, i].squeeze().tolist())

        # Based on Datacomp-clip, we only want to exclude the sample
        # if any of the category scores exceeds a threshold of 0.1.
        # see appendix E in https://arxiv.org/pdf/2304.14108.
        return max(results) <= 0.1


    #### 14/04 (Vishaal): this is the batched version of the
    #### above function, this is what we use!!
    @torch.no_grad()
    def predict_many_batched(self, texts, chunk_size=512):
        """
        Truly batched version of predict_many.
        Returns a list of booleans indicating if each text is non-toxic (True) or toxic (False).
        """

        self.model.eval()

        all_input_ids = []
        all_attention_masks = []
        mapping = []  # maps each chunk to the original text index

        # 1. Tokenize without truncation
        for text_idx, text in enumerate(texts):
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                padding=False
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = torch.ones_like(input_ids)  # since we didn't use tokenizer's padding

            current_chunks = 0
            for i in range(0, len(input_ids), chunk_size):

                # only allow upto certain max chunk-size
                if current_chunks > MAX_CHUNKS:
                    break

                chunk_ids = input_ids[i : i + chunk_size]
                chunk_mask = attention_mask[i : i + chunk_size]

                pad_len = chunk_size - chunk_ids.shape[0]
                if pad_len > 0:
                    chunk_ids = torch.nn.functional.pad(chunk_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
                    chunk_mask = torch.nn.functional.pad(chunk_mask, (0, pad_len), value=0)

                all_input_ids.append(chunk_ids)
                all_attention_masks.append(chunk_mask)
                mapping.append(text_idx)

                current_chunks += 1

        # 2. Stack all chunks
        input_ids_batch = torch.stack(all_input_ids).to(self.model.device)  # [N_chunks, 512]
        attention_mask_batch = torch.stack(all_attention_masks).to(self.model.device)

        # 3. Single forward pass

        with autocast(device_type='cuda', enabled=(self.device == "cuda")):
            logits = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
            scores = torch.sigmoid(logits).cpu()  # [N_chunks, num_labels]

        # 4. Aggregate max score per text
        num_texts = len(texts)
        max_scores = [torch.full((len(self.class_names),), -float("inf")) for _ in range(num_texts)]

        for i, idx in enumerate(mapping):
            max_scores[idx] = torch.max(max_scores[idx], scores[i][self.class_indices])

        # 6. Apply threshold rule
        return [torch.all(score <= 0.1).item() for score in max_scores]


def log_and_continue(exn):
    # removing this to ensure we don't print on terminal, gets annoying to look at!
    # print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
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

def decode_text_only(sample):
    for k in sample.keys():
        if 'txt' in k:
            sample[k] = sample[k].decode('utf-8')
    return sample

def test(input_folder: str, output_folder: str, filter_name: str, start_tar_index: str):
    print("Running tests (may take a while...)")
    
    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)
    random.shuffle(tarfiles)

    # run the test on a single random tar
    if start_tar_index is None:
        input_tarfile = osp.join(input_folder, osp.basename(tarfiles[0]))
        output_tarfile = osp.join(output_folder, osp.basename(tarfiles[0]))
    else:
        input_tarfile = osp.join(input_folder, '{}.tar'.format(start_tar_index))
        output_tarfile = osp.join(output_folder, '{}.tar'.format(start_tar_index))
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

        # make sure that new value added is binary
        assert filters[filter_name] in [0, 1], "The detoxify scoring should be binary"

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


def write_batch_to_tar(batch, fds, output_folder, writers, iterators):
    # once we have the processed vals, we do 
    # sequential writing of tars
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
        enc_sample["filters.json"] = fds[index]

        # get tar writer
        if lp_ not in writers:
            # get path to written tar
            ot_ = osp.join(output_folder, osp.basename(lp_))
            # we create writer
            writers[lp_] = TarWriter(ot_)

        # write the sample
        writers[lp_].write(enc_sample)


def main(args):

    # flat list of tarfiles in the current directory
    tarfiles = glob(osp.join(args.input_folder, "*.tar"), recursive=False)
    # sort them based on the integer key from the filename
    tarfiles = sorted(
        tarfiles,
        key=lambda x: int(osp.basename(x).split(".")[0])
    )

    # create write executor once on a single separate thread
    writer_executor = None
    if args.use_thread_writer:
        writer_executor = ThreadPoolExecutor(max_workers=1)

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

    # get detoxify model
    model_ = Detoxify('multilingual', device=args.device)

    # get wds pipeline
    decoded_dataset = wds.DataPipeline(
        wds.SimpleShardList(brace_expanded),  # this will give an iterator over all tars
        wds.split_by_worker,  # this will distribute across all cpu workers within a rank (num_workers per gpu)
        tarfile_to_samples_throw_wrapped,  # this will give us sample dicts per worker
        wds.map(decode_text_only, handler=log_and_continue), # since we only operate on the texts, we save time for image decoding by only decoding texts
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

    tot_samps = 0
    tot_removed = 0

    # iter through tars
    for batch in tqdm(dataloader):

        # we first get the full batch in correct format
        full_seqs = []
        for dec_sample in batch:
            # get all texts concatenated in the sequence
            full_seq = ""
            for k, v in dec_sample.items():
                # skip if its json file or image or metadata
                if k[:2] == "__":
                    continue
                if not k.endswith("txt"):
                    continue
                # ensure its a text
                assert isinstance(v, str)
                full_seq += v + " "
            if len(full_seq) > 1:
                full_seq = full_seq[:-1]
            full_seqs.append(full_seq)

        # now we run the full model on the formatted batch
        preds = model_.predict_many_batched(full_seqs)

        # get the filters dictionary for each sample and write the filter val
        fds = []
        for dec_sample, pred in zip(batch, preds):
            filters_dict = dec_sample.get("filters.json", {})
            tot_samps += 1
            if not pred:
                tot_removed += 1
            filters_dict[args.filter_name] = 0 if pred==False else 1
            fds.append(filters_dict)

        # once we have the processed vals, we do 
        # sequential writing of tars
        if args.use_thread_writer:
            # submit async writing task
            writer_executor.submit(write_batch_to_tar, batch, fds, output_folder, writers, iterators)
        else:
            # write in main process
            write_batch_to_tar(batch, fds, output_folder, writers, iterators)

    # After loop ends: wait for all pending writes to finish if done async
    if writer_executor is not None:
        writer_executor.shutdown(wait=True)

    # close all writer streams
    for lp_ in writers:
        writers[lp_].close()    

    print("Took {} seconds to write everything!".format(time.time()-st))
    print("Removed: {}, Total: {}, Percent removed: {}".format(tot_removed, tot_samps, 100. * tot_removed/tot_samps))

    # return paths for testing
    return args.input_folder, output_folder



if __name__ == "__main__":
    # very simple cmd-line args, feel free to extend
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--start_tar_index', type=str, required=False, default=None, help="If set, the start index in full string format (eg, '00000') to add in the brace-expanded string of webdataset, else will automatically construct the brace-expanded string")
    parser.add_argument('--end_tar_index', type=str, required=False, default=None, help="If set, the end index in full string format (eg, '00100') to add in the brace-expanded string of webdataset, else will automatically construct the brace-expanded string")
    parser.add_argument('--use_thread_writer', action='store_true', help='If set, use a background thread to write tar samples asynchronously.')
    parser.add_argument('--filter_name', type=str, default="detoxify")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # filtering here
    input_folder, output_folder = main(args)

    # run a simple unit-test
    test(input_folder, output_folder, args.filter_name, args.start_tar_index)


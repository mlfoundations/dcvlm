"""
This utility scripts converts instruction-tuning datasets in .jsonl format into datasets to be consumed by WebDataset.

The format is as follows:

1. Each jsonl line constitutes a sample;

2. All images related to a line are encoded into byte-strings;
- for single-image cases, images are encoded under the "jpg" key, following the format of img2dataset to store image-text pairs.
- for multi-image samples, these are encoded as "0.jpg", "1.jpg", "2.jpg", and so on, following the format of multimodal documents;

3. The `conversations` list, which is the core of the instruction-tuning format, is converted into a raw string and stored
as the 'conversations.txt' key of each sample; This implies that text-only samples will not have any "jpg" key, but will only
have the 'conversations.txt', which can be converted back to the original conversations list
using the txt2conversations function.

The script supports passing some metadata keys, which you want to remove from the jsonl line before json-encoding, to
store them into a different destination field (which by default is the "filters.json" key). This is useful if, for example,
one wants to store metadata from filtering scripts that have been executed already into the destination WebDatasets, without losing
access to them.
"""

import io
import os
import time
import json
import torch
import subprocess
import os.path as osp
import multiprocessing as mp

from glob import glob
from tqdm import tqdm
from datetime import timedelta
from webdataset import TarWriter
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed

# special character used to delimit
# conversation turns, allows for lossless decoding
EOL_SEPARATOR = "<EOCL>\n"

# keys with injected metadata of the filters we ran so far
META_KEYS = [
    "detoxify",
    "length_chars",
    "length_tokens",
    "length_tokens_text_only",
    "length_tokens_trainable_only",
    "length_images",
    "textdecontamination",
    "iqa_live_mean",
    "iqa_live_min",
    "iqa_live_max",
    "iqa_csiq_mean",
    "iqa_csiq_min",
    "iqa_csiq_max",
    "iqa_tid2013_mean",
    "iqa_tid2013_min",
    "iqa_tid2013_max",
    "iqa_kadid10k_mean",
    "iqa_kadid10k_min",
    "iqa_kadid10k_max",
    "iqa_flive_mean",
    "iqa_flive_min",
    "iqa_flive_max",
    "iqa_spaq_mean",
    "iqa_spaq_min",
    "iqa_spaq_max",
    "iqa_clive_mean",
    "iqa_clive_min",
    "iqa_clive_max",
    "iqa_koniq10k_mean",
    "iqa_koniq10k_min",
    "iqa_koniq10k_max"
]


# mapping of images descriptor keys,
# will use them for train/test decontamination
DESC_KEYS = {
    "p_hash": "phash.txt",
    "sha256": "sha256.txt",
}

# list of executed filters so far
# this filters .jsonl files within a root folder
EXECUTED_FILTERS = [
    "detoxify",
    "sscd",
    "textdecontamination",
    "length",
    "iqa"
]

# define a limit on the shared queue to avoid OOM issues
BASE_QUEUED_SAMPLES = 10_000
MAX_QUEUED_SAMPLES = 80_000


def emb2bytes(embedding: torch.Tensor) -> bytes:
    "Convert a raw PyTorch tensor to a byte string"
    buffer = io.BytesIO()
    torch.save(embedding, buffer)
    buffer.seek(0)
    return buffer.read()


def conversations2txt(conversations: list[dict]) -> str:
    "Transform a list of conversation turns into a raw string"
    conversation_text = ""
    for turn_idx, turn in enumerate(conversations):
        role = turn["from"]
        value = turn["value"]
        end_of_line = EOL_SEPARATOR if (turn_idx + 1) < len(conversations) else ""
        conversation_text += f"{role}:{value}{end_of_line}"
    return conversation_text


def txt2conversations(txt: str) -> list[dict]:
    "Transform a raw string into a list of conversation turns"
    conv_lines = txt.split(EOL_SEPARATOR)
    conversations = []
    for line in conv_lines:
        # by construction, the 'role' is prepended
        # to the actual value, followed by a column
        role = line.split(":")[0]

        # by construction, the rest of the line is the value
        # (the +1 is to remove the manually added column)
        value = line[(len(role)+1) :]

        # we can then use role and value to create the dict for this turn
        turn = {"from": role, "value": str(value)}
        conversations.append(turn)
    return conversations


def load_jsonl(data_root: str, metadata_dir: str) -> list[dict]:

    def count_lines(filepath: str) -> int:
        result = subprocess.run(
            ["wc", "-l", filepath], 
            stdout=subprocess.PIPE, 
            check=True
        )
        return int(result.stdout.decode("utf-8").strip().split()[0])
    
    # list the directory
    jsonl_files = glob(osp.join(data_root, "*.jsonl"))

    # keep only jsonl files from successfully executed filters
    jsonl_files = sorted([
        f for f in jsonl_files 
        if all([filt in f for filt in EXECUTED_FILTERS])
    ])
    log_str = "\n".join(jsonl_files)
    print(f"Found {len(jsonl_files)} JSONL files to load:\n{log_str}")

    # check if total lines file exists
    totlines_file = osp.join(metadata_dir, "totlines.txt")
    if osp.exists(totlines_file):
        with open(totlines_file, "r") as f:
            totlines = int(f.read().strip())
        print(f"Total lines file found, skipping line counting. Total lines: {totlines:,}")
        return jsonl_files, totlines
    
    # compute total number of lines
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        totlines = 0
        for jf in jsonl_files:
            futures.append(
                executor.submit(count_lines, jf)
            )
        for future in tqdm(
            as_completed(futures), 
            total=len(jsonl_files), 
            desc="Computing total lines"
        ):
            totlines += future.result()
    
    # write the number of line to a file,
    # so when resuming we can skip this step
    with open(totlines_file, "w") as f:
        f.write(str(totlines))

    return jsonl_files, totlines


def line2sample(
    line: dict,
    data_root: str,
    metadata_keys: list = META_KEYS,
    dest_key: str = "filters.json",
) -> dict:

    sample = {"__key__": str(line["id"])}

    # encode images into image-bytes
    if "image" in line:
        # - single-image cases
        if isinstance(line["image"], str):
            with open(osp.join(data_root, line["image"]), "rb") as img_in:
                img = img_in.read()
            sample["jpg"] = img

        elif isinstance(line["image"], list) and len(line["image"]) == 1:
            with open(osp.join(data_root, line["image"][0]), "rb") as img_in:
                img = img_in.read()
            sample["jpg"] = img

        # - multi-image case
        elif isinstance(line["image"], list) and len(line["image"]) > 1:
            for img_idx, img_base in enumerate(line["image"]):
                with open(osp.join(data_root, img_base), "rb") as img_in:
                    img = img_in.read()
                sample[f"{img_idx}.jpg"] = img

    # prepare conversations text
    src_conv = line["conversations"]
    txt_conv = conversations2txt(src_conv)
    tmp_conv = txt2conversations(txt_conv)
    # make sure the reformatted conversation can be brought back to its 
    # original state
    def _string_cvt(conv: list[dict]):
        new_conv = []
        for line in conv:
            if line.get("from", None) is not None:
                line['from'] = str(line.get("from"))
            if line.get("role", None) is not None:
                line['role'] = str(line.get("role"))
            if line.get("value", None) is not None:
                line['value'] = str(line.get("value"))
            new_conv.append(line)
        return new_conv
    assert tmp_conv == _string_cvt(src_conv), f"Unable to decode conversation. Got:{tmp_conv}, should've been:{src_conv}"
    sample["conversations.txt"] = txt_conv.encode("utf-8")

    # save all metadata keys into the destination key of the sample
    sample[dest_key] = {}
    for meta_key in metadata_keys:
        try:
            sample[dest_key][meta_key] = line[meta_key]
        except KeyError as ke:
            # acceptable if textdecont is missing from image samples
            if meta_key == 'textdecontamination' and "image" in line: 
                continue
            # acceptable if iqa scores are missing when no images are there
            elif "iqa" in meta_key and "image" not in line: 
                continue
            # FIXME: patch to assign a default value for missing textdecontamination
            elif meta_key == 'textdecontamination' and "image" not in line:
                sample[dest_key][meta_key] = 1
            # otherwise, raise
            else:  
                raise KeyError(ke)

    sample[dest_key] = json.dumps(sample[dest_key]).encode("utf-8")

    # now we move over also image descriptors for train/test
    # decontamination and dedup
    for desc_jsonl_key, desc_wds_key in DESC_KEYS.items():
        # text datasets will not have image descriptors
        if desc_jsonl_key not in line: continue
        
        # handle single- and multi-image cases
        if isinstance(line[desc_jsonl_key], str):
            sample[desc_wds_key] = line[desc_jsonl_key].encode("utf-8")
        elif isinstance(line[desc_jsonl_key], list):
            for idx, descriptor in enumerate(line[desc_jsonl_key]):
                sample[f"{idx}.{desc_wds_key}"] = descriptor.encode("utf-8")

    # finally, image embeddings computed with the sscd model 
    # need special treatment
    if "sscd_embedding" in line:
        # single embedding cases
        if isinstance(line['sscd_embedding'], str):
            embedding = torch.load(
                osp.join(data_root, line['sscd_embedding']),
                weights_only=True
            )
            assert embedding.dtype == torch.float16
            sample["sscd_disc_mixup.pt"] = emb2bytes(embedding)

        elif isinstance(line['sscd_embedding'], list) and len(line['sscd_embedding']) == 1:
            embedding = torch.load(
                osp.join(data_root, line['sscd_embedding'][0]),
                weights_only=True
            )
            assert embedding.dtype == torch.float16
            sample["sscd_disc_mixup.pt"] = emb2bytes(embedding)
        
        # multiple embeddings case
        if isinstance(line['sscd_embedding'], list) and len(line['sscd_embedding']) > 1:
            for idx, emb_path in enumerate(line['sscd_embedding']):
                embedding = torch.load(
                    osp.join(data_root, emb_path),
                    weights_only=True
                )
                assert embedding.dtype == torch.float16
                sample[f"{idx}.sscd_disc_mixup.pt"] = emb2bytes(embedding)

    # encode samples
    return sample


def producer(files_queue: mp.Queue, samples_queue: mp.Queue):
    "Loads jsonl files and puts decoded lines in a shared queue"
    while True:
        inp_file = files_queue.get()
        if inp_file == "done": 
            break

        # open the file, start reading lines and put them in a shared queue
        with open(inp_file, "r", encoding="utf-8") as f:
            # put lines into the queue
            for line in f:
                try:
                    jline = json.loads(line.strip())
                    samples_queue.put(jline)
                except Exception as e:
                    # if we cannot parse the line, put the exception in the queue
                    # so that it can be handled by the consumer
                    samples_queue.put(e)
    
    print("End of producer process.")



def consumer(
        args: Namespace, 
        samples: mp.Queue, 
        success: mp.Queue,
        shard_lock, 
        error_lock,
        shard_counter,
        total_shards,
        shard_sizes
    ):
    
    # local fn to convert a json line to a sample
    def _load_sample(line):
        try:
            if isinstance(line, Exception):
                raise line
            # convert the json line to a sample
            sample = line2sample(line, args.data_root)
            return sample
        except Exception as e:
            return e

    # within-process sample counter
    local_consumed = 0
    consumed = 0
    written = 0
    init_time = time.time()

    # initialize a writer before starting
    with shard_lock:
        outpath = osp.join(args.output_dir, f"{str(shard_counter.value).zfill(5)}.tar")
        writer = TarWriter(outpath)
        samples_per_shard = shard_sizes[shard_counter.value]
        shard_counter.value += 1

    # start dumping samples
    while True:
        
        # pull samples from the queue
        sample = samples.get()
        if sample == "done": 
            break
        sample = _load_sample(sample)
        
        # write the sample on disk
        if not isinstance(sample, Exception):
            writer.write(sample)
            written += 1
        # otherwise, save the error to a file 
        # for offline diagnosis
        else:
            error_file = osp.join(args.metadata_dir, "errors.txt")
            with error_lock, open(error_file, "a") as ef:
                ef.write(f"{str(sample)}\n")
        
        local_consumed += 1
        consumed += 1
        
        # create new writer every :samples_per_shard: items
        if local_consumed == samples_per_shard:
            # close the previous writer
            writer.close()
            local_consumed = 0
            
            # grab the shared counter and increment it
            with shard_lock:
                # if we get here, somebody is already writing the last shard
                # so we can safely exit this consumer process
                if shard_counter.value >= total_shards:
                    break
                
                # create a new writer for the next shard
                outpath = osp.join(args.output_dir, f"{str(shard_counter.value).zfill(5)}.tar")
                writer = TarWriter(outpath)
                samples_per_shard = shard_sizes[shard_counter.value]
                shard_counter.value += 1

                if (
                    shard_counter.value % args.log_freq == 0
                    or shard_counter.value == total_shards
                ):
                    elapsed = int(time.time() - init_time)
                    time_per_shard = elapsed / shard_counter.value 
                    eta = int(time_per_shard * (total_shards - shard_counter.value))
                    print(
                        f"Opening shard {shard_counter.value:,} of {total_shards:,}.",
                        f"Successfully written (local): {written:,}.",
                        f"Consumed (local): {consumed:,}.",
                        f"Time elapsed: {timedelta(seconds=elapsed)}.",
                        f"ETA: {timedelta(seconds=eta)}.",
                        sep=" ",
                        flush=True
                    )

    # put successful results in the queue (dumping metadata)
    success.put({
        "consumed": consumed,
        "written": written
    })
    print("End of consumer process.")


def main(args):
    # time stats
    st = time.time()   

    # create metadata dir
    args.metadata_dir = osp.join(args.output_dir, "metadata")
    os.makedirs(args.metadata_dir, exist_ok=True)

    # if metadata dir exists, and stats file says that we already processed
    # all lines, we can skip the processing
    stats_file = osp.join(args.metadata_dir, "stats.txt")
    if osp.exists(stats_file):
        with open(stats_file, "r") as f:
            stats_lines = [l.strip() for l in f.readlines()]
        processed_lines = 0
        written_samples = 0
        for line in stats_lines:
            if "Total lines processed" in line:
                processed_lines = int(line.split(":")[-1].strip())
            if "Total samples written" in line:
                written_samples = int(line.split(":")[-1].strip())
        if written_samples >= int(0.95*processed_lines):
            print(
                f"Metadata directory {args.metadata_dir} already exists.",
                "More than 95% of all lines have been processed successfully.",
                "Skipping."
            )
            return
    
    # create output dir with filter names
    args.output_dir = osp.join(args.output_dir, "shards", *EXECUTED_FILTERS)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Will save shards at {args.output_dir}.")
    
    # remove existing error file
    if osp.isfile(osp.join(args.metadata_dir, "errors.txt")):
        os.remove(osp.join(args.metadata_dir, "errors.txt"))

    # grab all jsonl files to convert
    jsonl_files, total_lines = load_jsonl(data_root=args.data_root, metadata_dir=args.metadata_dir)
    
    # distribute lines evenly across shards
    total_shards = max(1, round(total_lines / args.samples_per_shard))
    base_shard_size = total_lines // total_shards
    remainder = total_lines % total_shards
    sizes = [base_shard_size+1]*remainder + [base_shard_size]*(total_shards-remainder)
    print(f"Total shards to create: {total_shards:,}.")

    # define producers and consumers
    num_producers = min(len(jsonl_files), args.num_producers)
    num_consumers = min(total_shards, args.num_consumers)
    num_consumers = max(num_consumers, num_producers)
    
    # define shared items across processes
    shard_counter = mp.Value("i", 0) 
    shard_lock = mp.Lock()
    error_lock = mp.Lock()

    success_queue = mp.Queue() # for metadata
    files_queue = mp.Queue()  # jsonl paths
    max_queued_samples = min(
        BASE_QUEUED_SAMPLES * num_consumers,
        MAX_QUEUED_SAMPLES
    )
    samples_queue = mp.Queue(maxsize=max_queued_samples)  # converted samples
    
    # put jsonl files into the files queue
    for jf in jsonl_files:
        files_queue.put(jf)
    
    # ensure each producer has a sentinel
    for _ in range(num_producers):
        files_queue.put("done")

    # start producer processes 
    # each producer reads a jsonl file 
    # converts lines and puts them in a shared queue
    print(f"Starting {num_producers} producer processes.")
    producers = []
    for _ in range(num_producers):
        producer_proc = mp.Process(
            target=producer, 
            args=(files_queue, samples_queue)
        )
        producer_proc.start()
        producers.append(producer_proc)

    # start consumer procs to write samples on disk
    # each consumer pops from the queue, and writes to disk
    # on separate shards
    args.log_freq = max(args.log_freq, num_consumers)
    print(f"Starting {num_consumers} consumer processes.")
    consumers = []
    for _ in range(num_consumers):
        consumer_proc = mp.Process(
            target=consumer, 
            args=(
                args, 
                samples_queue, 
                success_queue, 
                shard_lock, 
                error_lock, 
                shard_counter, 
                total_shards,
                sizes
            )
        )
        consumer_proc.start()
        consumers.append(consumer_proc)
    
    # end gracefully
    for p in producers:
        p.join()
    
    # ensure each consumer has a sentinel
    for _ in range(num_consumers):
        samples_queue.put("done")
    
    # and gracefully terminate consumers
    for p in consumers:
        p.join()

    # collect results from consumers
    total_consumed = 0
    total_written = 0
    while not success_queue.empty():
        res = success_queue.get()
        total_consumed += res["consumed"]
        total_written += res["written"]

    # dump stats to a file
    with open(osp.join(args.metadata_dir, "stats.txt"), "w") as f:
        f.write(f"Total samples consumed: {total_consumed}\n")
        f.write(f"Total samples written: {total_written}\n")
        f.write(f"Total shards created: {shard_counter.value}\n")
        f.write(f"Total lines processed: {total_lines}\n")

    # log total time and exit
    total_time = int(time.time()-st)
    print(f"Done! Total time: {timedelta(seconds=total_time)}.")
    print(f"Total samples consumed: {total_consumed:,}.")
    print(f"Total samples written: {total_written:,}.")




if __name__ == "__main__":
    parser = ArgumentParser()
    # input and output directories
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples_per_shard", type=int, default=10_000)
    # params for multiprocessing
    parser.add_argument("--num_producers", type=int, default=8)
    parser.add_argument("--num_consumers", type=int, default=8)
    # misc
    parser.add_argument("--log_freq", type=int, default=10)
    args = parser.parse_args()
    main(args)

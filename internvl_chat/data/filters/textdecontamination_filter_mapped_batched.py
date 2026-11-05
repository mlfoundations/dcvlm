'''
Script to do exact search for text-decontamination with MinHash signatures.

Full Description:
A script to run decontamination against a single jsonl file. This script utilizes a two-stage 
decontamination procedure: (1) If the Jaccard similarity between the Minhash signatures between 
a train and the closest test sample is greater than 0.55 (determined via a small human annotation 
experiment conducted by us), we mark the sample as contaminated, and (2) For samples that have 
Jaccard similarity between 0.3 and 0.55, we manually check if there is an exact substring match 
(bi-directionally) between the train and the closest matched test sample (i.e. if either the train 
sample is entirely contained in the test sample or vice versa).

Note: for now we do not consider approximate LSH matching since exact search is fast enough for us.
'''
import os
import json
import random
from tqdm import tqdm
import re
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from argparse import Namespace
import pickle
from datasketch import MinHash, MinHashLSH
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader

import time

TOKENIZER_PATTERN = r"[ ,.?!:;()\"-]|\n|\\"  # tokenizer pattern used in Tulu-3 decont pipeline, NOTE: this might not work well with Chinese text!!

class JSONLConversationDataset(Dataset):
    def __init__(self, filepath):
        
        # load the full dataset in-memory
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        # since we do n-gram matches simply concatenate the user parts of the conversation
        text = ' '.join(str(d["value"]) for d in line["conversations"] if d is not None and d["from"]=="human")
        return line, text

def collate_fn(batch):
    lines, texts = zip(*batch)
    return list(lines), list(texts)

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as file_ptr:
        data = [json.loads(line) for line in file_ptr]
    return data

def make_output_path(args: Namespace):
    assert isinstance(args.input_file, str) \
        and args.input_file.endswith(".jsonl") \
            and args.input_file.count(".jsonl") == 1
    return args.input_file.replace(".jsonl", f"_{args.filter_name}.jsonl")

def custom_tokenize(text):
    # We implement this custom-tokenize func
    # to exactly match the tokenization used in
    # Tulu-3 decontamination ES-pipeline.
    # see https://github.com/allenai/open-instruct/blob/4ce14fe5506c3e900214805a375ae1c204edee4e/decontamination/index.py#L10-L32
    # However, note that the Tulu-3 tokenizer
    # does not work well for Chinese, this is
    # a limitation we have to be aware of!!
    text = text.lower()
    tokens = re.split(TOKENIZER_PATTERN, text)
    return [t for t in tokens if t]

def batch_compute_minhashes(docs, n=8, num_perm=128):
    # this function uses 8-grams by default as that's what's used in Tulu-3 (sec 3.2 https://arxiv.org/pdf/2411.15124)
    all_ngrams_encoded = []
    all_texts_encoded = []
    total = 0
    encoded_mask = []  # construct a mask whether each sample has a valid encoding or not
    for text in docs:
        total += 1
        tokens = custom_tokenize(text)
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if len(ngrams) < 1:
            # print("Warning: Skipping sample! You have set too high n={} threshold, cannot create n-gram sequence for this sample: {}\n\n\n\n".format(n, text))
            encoded_mask.append(0)  # since we are skipping this sample, we don't have a valid encoding!
            continue
        encoded = [ng.encode('utf-8') for ng in ngrams]
        assert len(encoded) >= 1, "There is something wrong, the number of elements in this sample's n-gram set is less than 1, sample: {}".format(text)
        all_ngrams_encoded.append(encoded)
        all_texts_encoded.append(text)
        encoded_mask.append(1)  # we have a valid encoding for this sample!
    return MinHash.bulk(all_ngrams_encoded, num_perm=num_perm), all_texts_encoded, encoded_mask

def compute_matches(index, queries, device, batch_size=128):
    '''
    Performs exact nearest neighbor search over MinHash signatures using Jaccard similarity.

    This function compares each query MinHash signature against a provided index (dictionary of MinHashes),
    computes the pairwise Jaccard similarity using signature matching, and returns:

        - top1_scores_all: the Jaccard similarity score of the closest match for each query
        - top1_indices_all: the index (position) in the index dictionary of the closest match for each query
        - latency: total time (in seconds) taken for the search over all queries

    The search is batched and runs fully on the specified `device` (e.g., 'cuda' or 'cpu').

    Args:
        index (dict): Dictionary mapping keys to MinHash signatures.
        queries (list): List of MinHash query signatures.
        device (str or torch.device): Device to perform computation on.
        batch_size (int, optional): Number of queries to process per batch. Default is 128.

    Returns:
        top1_scores_all (np.ndarray): Array of top Jaccard similarity scores (shape: [num_queries]).
        top1_indices_all (np.ndarray): Array of indices in the index corresponding to top matches (shape: [num_queries]).
        latency (float): Total time taken (in seconds) for computing all matches.
    '''

    # assert the types
    assert isinstance(index, dict)
    assert isinstance(queries, list)

    # get np index
    index_signatures_np = np.stack([v for v in index.values()])  # (N, D)
    index_signatures = torch.from_numpy(index_signatures_np).to(device)  # (N, D)

    num_queries = len(queries)
    top1_indices_all = np.empty(num_queries, dtype=np.int32)
    top1_scores_all = np.empty(num_queries, dtype=np.float32)

    # get similarities
    st = time.time()
    for batch_index in range(0, num_queries, batch_size):

        # get ranges
        end_index = min(batch_index + batch_size, num_queries)
        batch_range = slice(batch_index, end_index)

        # get np bs-index
        rand_sigs_np = np.stack([r.digest() for r in queries[batch_range]])  # (bs, D)
        rand_sigs = torch.from_numpy(rand_sigs_np).to(device)  # (bs, D)

        rand_expand = rand_sigs[:, None, :]  # (bs, 1, D)
        index_expand = index_signatures[None, :, :]  # (1, N, D)

        matches = (rand_expand == index_expand)  # shape: (bs, N, D)

        counts = matches.sum(dim=2)

        similarity_matrix = counts.float() / rand_sigs.shape[1]  # (bs, N)

        top1_scores, top1_indices = similarity_matrix.max(dim=1)  # (bs,)

        # Save results to final arrays
        top1_scores_all[batch_range] = top1_scores.cpu().numpy()
        top1_indices_all[batch_range] = top1_indices.cpu().numpy()

    return top1_scores_all, top1_indices_all, time.time()-st

def test(input_file: str, output_file: str, filter_name: str, removed: int, delete_old: bool):
    """
    Use this function as a final "unit-test" for your script (may take some time if your .jsonl is large). 
    :param input_file: path to input jsonl file;  
    :param output_file: path to output jsonl file:
    """
    
    # load input and output .jsonl files
    input_data = load_jsonl(input_file) 
    output_data = load_jsonl(output_file)

    # no lines should be added or removed
    assert len(output_data) == len(input_data)

    # files should be row-wise identical except for the new key
    removed_vals = 0
    for row_in, row_out in zip(input_data, output_data):

        # make sure new key is there
        assert filter_name in row_out

        # make sure that new value added is binary
        assert row_out[filter_name] in [0, 1]
        if row_out[filter_name] == 0:
            removed_vals += 1

        # make sure other values did not change
        for k, v in row_in.items():
            assert row_out[k] == v, \
                f"Value at key {k} changed from {v} to {row_out[k]}"
            
    assert removed_vals == removed, "Total removed tracked in the file is different from the one tracked in script"

    print("Test ran through with no assertion errors =)")

    if delete_old:
        print("Deleting the input file {}!!".format(input_file))
        os.remove(input_file)

def main(args):

    # get time stats
    st = time.time()

    # ensure text index exists and load it   
    assert os.path.exists(args.index_path) and os.path.isfile(args.index_path)
    with open(args.index_path, 'rb') as f:
        index = pickle.load(f)
    map_path = args.index_path.removesuffix(".index")+".inds2samples"
    with open(map_path, 'rb') as f:
        inds2samples = pickle.load(f)
    assert isinstance(index, dict)
    assert isinstance(inds2samples, list)

    output_file = make_output_path(args)

    dataset = JSONLConversationDataset(args.input_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.data_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    tot_samps = 0
    tot_removed = 0
    tot_removed_first_stage = 0
    tot_removed_second_stage = 0

    with open(output_file, "w", encoding="utf-8") as out_ptr:
        for batch_lines, batch_texts in tqdm(dataloader, total=len(dataloader), ascii=True):

            # first get the full set of signatures for current batch
            current_signatures, current_train_samples, encoded_mask = batch_compute_minhashes(batch_texts, n=args.n_gram_threshold, num_perm=args.signature_size)
            assert len(current_signatures) == sum(encoded_mask)
            assert len(encoded_mask) == len(batch_texts)

            # compute the matches for the current batch
            top1_scores, top1_indices, _ = compute_matches(index, current_signatures, args.device, args.model_batch_size)

            # compute binary hit vector
            hits = top1_scores >= args.sim_threshold  # shape: (num_queries,)
            lenient_hits = top1_scores >= args.lenient_threshold  # shape: (num_queries,)

            assert len(hits) == len(current_train_samples)
            assert len(lenient_hits) == len(current_train_samples)

            # write each new output line
            matched = 0
            for mask_val, line in zip(encoded_mask, batch_lines):
                tot_samps += 1

                # if mask value is 0, then we keep the sample as is without discarding since we could not get a valid encoding
                if mask_val == 0:
                    line[args.filter_name] = 1
                    out_ptr.write(json.dumps(line) + "\n")
                    continue

                # if mask value is 1, we should load the match
                is_hit = hits[matched]

                # if it is a hit then set the filter value to be 0
                # if it is not a hit, do a second stage where you get the closest text sample
                # and then do an exact search
                if is_hit:
                    line[args.filter_name] = 0
                    tot_removed += 1
                    tot_removed_first_stage += 1

                    # get the train and the test candidates, and lowercase them to improve hit recall
                    train_samp = current_train_samples[matched].lower()
                    test_samp = inds2samples[top1_indices[matched]].lower()

                    # if we want to visualize all matched, print them!
                    if args.vis_hits:
                        print("_________")
                        print("First-Stage Hit")
                        print(train_samp)
                        print(test_samp)
                        print("_________")
                        print()

                else:
                    is_second_stage_hit = lenient_hits[matched]
                    if is_second_stage_hit:

                        # get the train and the test candidates, and lowercase them to improve hit recall
                        train_samp = current_train_samples[matched].lower()
                        test_samp = inds2samples[top1_indices[matched]].lower()

                        # if they are exact substrings of each other, match them
                        if train_samp in test_samp or test_samp in train_samp:
                            line[args.filter_name] = 0
                            tot_removed += 1
                            tot_removed_second_stage += 1

                            # if we want to visualize all matched, print them!
                            if args.vis_hits:
                                print("_________")
                                print("Second-Stage Hit")
                                print(train_samp)
                                print(test_samp)
                                print("_________")
                                print()

                        else:
                            line[args.filter_name] = 1
                    else:
                        line[args.filter_name] = 1

                matched += 1
                out_ptr.write(json.dumps(line) + "\n")

    print("Took {} seconds to write everything!".format(time.time()-st))
    print("Removed: {}, Removed first-stage: {}, Removed second-stage: {}, Total: {}, Percent removed: {}".format(tot_removed, tot_removed_first_stage, tot_removed_second_stage, tot_samps, 100. * tot_removed/tot_samps))

    return args.input_file, output_file, tot_removed


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--index_path', type=str, required=True, help="Where to load the MinHash-Exact index from.")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--data_batch_size', type=int, default=2048)
    parser.add_argument('--model_batch_size', type=int, default=128)
    parser.add_argument('--filter_name', type=str, default="textdecontamination")
    parser.add_argument('--n_gram_threshold', type=int, default=5)
    parser.add_argument('--sim_threshold', type=float, default=0.55)
    parser.add_argument('--lenient_threshold', type=float, default=0.3)
    parser.add_argument('--signature_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--vis_hits', action="store_true", help="whether to print all the contaminated samples in the train set (along with the matched test samples).")
    parser.add_argument('--delete_old', action="store_true", help="whether to delete the input_file and only keep the output_file")
    args = parser.parse_args()

    assert args.lenient_threshold <= args.sim_threshold, "The second-stage threshold should be smaller than the main threshold!"

    # filtering here
    input_file, output_file, rmd = main(args)

    # run a simple unit-test
    test(input_file, output_file, args.filter_name, removed=rmd, delete_old=args.delete_old)

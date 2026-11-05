import os
import json
import random

from argparse import ArgumentParser
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from tqdm import tqdm

class JSONLConversationDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        return line

def collate_fn(batch):
    return batch

def make_output_path(args: Namespace):
    assert isinstance(args.input_file, str) \
        and args.input_file.endswith(".jsonl") \
            and args.input_file.count(".jsonl") == 1
    return args.input_file.replace(".jsonl", f"_{args.filter_name}.jsonl")

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as file_ptr:
        data = [json.loads(line) for line in file_ptr]
    return data

def test(input_file: str, output_file: str, filter_name: str):
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
    for row_in, row_out in zip(input_data, output_data):

        # make sure new key is there
        assert filter_name in row_out

        # make sure other values did not change
        for k, v in row_in.items():
            assert row_out[k] == v, \
                f"Value at key {k} changed from {v} to {row_out[k]}"
            
    print("Test ran through with no assertion errors =)")


def main(args):
    # example function to create your output file
    # feel free to change it
    output_file = make_output_path(args)

    # time stats
    st = time.time()

    # load data in batched manner
    dataset = JSONLConversationDataset(args.input_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # TODO: implement stuff here, writing to output_file
    # (this example just assigns a silly random score from 1 to 10)
    with open(output_file, "w") as out_ptr:
        for lines in tqdm(dataloader, total=len(dataloader), ascii=True):

            ## this is where you would do your
            ## batch operation for the filter
            rands = np.random.rand(len(lines))

            ## once we have all the filtered values
            ## we write sequentially into the jsonl
            for line, pred in zip(lines, rands):
                line[args.filter_name] = pred
                out_ptr.write(json.dumps(line) + "\n")

    print("Took {} seconds to write everything!".format(time.time()-st))

    # return paths for testing
    return args.input_file, output_file


if __name__ == "__main__":
    # very simple cmd-line args, feel free to extend
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--filter_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--prefetch_factor', type=int, default=6)
    args = parser.parse_args()
    
    # do your filtering here
    input_file, output_file = main(args)

    # and run a simple unit-test
    # NOTE: this may not cover all cases, especially row-wise key check 
    # but we can take it as a reference / starting point
    test(input_file, output_file, args.filter_name)




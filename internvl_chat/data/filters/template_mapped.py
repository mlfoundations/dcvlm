import os
import json
import random

from argparse import ArgumentParser
from argparse import Namespace
import time

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as file_ptr:
        data = [json.loads(line) for line in file_ptr]
    return data


def make_output_path(args: Namespace):
    assert isinstance(args.input_file, str) \
        and args.input_file.endswith(".jsonl") \
            and args.input_file.count(".jsonl") == 1
    return args.input_file.replace(".jsonl", f"_{args.filter_name}.jsonl")


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
    input_data = load_jsonl(args.input_file)

    # time stats
    st = time.time()

    # TODO: implement stuff here, writing to output_file
    # (this example just assigns a silly random score from 1 to 10)
    with open(output_file, "w") as out_ptr:
        for line in input_data:
            line[args.filter_name] = float(random.randint(1, 10))
            line_to_dump = json.dumps(line) + "\n"
            out_ptr.write(line_to_dump)

    print("Took {} seconds to write everything!".format(time.time()-st))

    # return paths for testing
    return args.input_file, output_file



if __name__ == "__main__":
    # very simple cmd-line args, feel free to extend
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--filter_name', type=str, required=True)
    args = parser.parse_args()
    
    # do your filtering here
    input_file, output_file = main(args)

    # and run a simple unit-test
    # NOTE: this may not cover all cases, especially row-wise key check 
    # but we can take it as a reference / starting point
    test(input_file, output_file, args.filter_name)




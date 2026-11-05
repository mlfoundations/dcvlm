import os
import argparse
import requests
import pandas as pd
import os.path as osp

from tqdm import tqdm
from functools import partial

from concurrent.futures import ThreadPoolExecutor, as_completed


def download(url: str, output_path: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f, tqdm(desc=osp.basename(output_path), total=total_size, unit="B", unit_scale=True, unit_divisor=1024, leave=True) as progress_bar:
        for chunk in response.iter_content(chunk_size=4096):
            f.write(chunk)
            progress_bar.update(len(chunk))


def line2file(line: pd.Series, root: str):
    bname, url = line['file_name'], line['cdn_link']
    output_path = osp.join(root, bname)
    # FIXME: (Matteo) definitely not the correct way to enable resuming, 
    # we should check for file-size as well 
    if not osp.exists(output_path):
        download(url, output_path)
    else:
        print(f"{output_path} exists already, skipping =)")

    return output_path


def parallel_download(lines, root):
    line2file_partial = partial(line2file, root=root)
    with ThreadPoolExecutor(max_workers=len(lines)) as executor: 
        submitted_items = {executor.submit(line2file_partial, line): line for line in lines}
        
        for future in as_completed(submitted_items):
            try:
                output_path = future.result()
                print(f"Downloaded: {output_path}")
            except Exception as e:
                print(f"Error downloading {submitted_items[future]}: {e}")



def main(args):
    # the links file is issued per user by Meta after accepting the SA-1B license
    # (https://ai.meta.com/datasets/segment-anything-downloads/)
    df = pd.read_csv(args.links_file, sep='\t')

    root = args.output_dir
    os.makedirs(root, exist_ok=True)

    processes = 8
    for i in range(0, len(df), processes):
        start, stop = i, min(i+processes, len(df))
        slice = df.iloc[start:stop]
        lines = [line for _, line in slice.iterrows()]
        parallel_download(lines, root=root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--links_file", type=str, required=True, help="SA-1B links file (tsv with file_name, cdn_link)")
    parser.add_argument("--output_dir", type=str, required=True, help="where to download the SA-1B tars")
    main(parser.parse_args())

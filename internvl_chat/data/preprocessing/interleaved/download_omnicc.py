import os
import random
import os.path as osp
import time
import argparse
from img2dataset import download

random.seed(0)


def main(args):
    all_months = os.listdir(args.parquet_root)
    url_list = [os.path.join(args.parquet_root, month, x) for month in all_months for x in os.listdir(os.path.join(args.parquet_root, month))]
    random.shuffle(url_list)
    url_list = url_list[args.start:args.end]

    # symlink the selected parquets into one folder (img2dataset needs a single path)
    parquets_dir = osp.join(args.output_dir, f"parquets_{args.start}_{args.end}")
    os.makedirs(parquets_dir, exist_ok=True)
    for x in url_list:
        symlink = osp.join(parquets_dir, osp.basename(x))
        if not osp.exists(symlink):
            os.symlink(x, symlink)

    output_folder = osp.join(args.output_dir, "shards")
    os.makedirs(output_folder, exist_ok=True)

    start = time.time()
    download(
        processes_count=16,
        thread_count=64,
        url_list=parquets_dir,
        disable_all_reencoding=True,
        output_folder=output_folder,
        output_format="webdataset",
        input_format="parquet",
        url_col="images",
        caption_col="texts",
        enable_wandb=True,
        number_sample_per_shard=10_000,
    )
    end = time.time()
    print(f"Time taken = {end-start} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_root", type=str, required=True, help="folder with the OmniCorpus-CC parquets (one subfolder per month)")
    parser.add_argument("--output_dir", type=str, required=True, help="where to write symlinked parquets and shards")
    parser.add_argument("--start", type=int, required=True, help="first parquet index (after a seeded shuffle)")
    parser.add_argument("--end", type=int, required=True, help="last parquet index, exclusive")
    main(parser.parse_args())

from img2dataset import download
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_list", type=str, required=True, help="WanJuan1.0 multi-modal *.jsonl.tar.gz file")
    parser.add_argument("--output_folder", type=str, required=True, help="where to write the shards")
    args = parser.parse_args()

    s = time.time()
    download(
        processes_count=8,
        thread_count=8,
        url_list=args.url_list,
        disable_all_reencoding=True,
        input_format="jsonl.tar.gz",
        output_folder=args.output_folder,
        output_format="webdataset",
        url_col="content",
        caption_col=None,
        enable_wandb=True,
        number_sample_per_shard=1000,
    )
    print(time.time() - s)
    print("Job finished!")

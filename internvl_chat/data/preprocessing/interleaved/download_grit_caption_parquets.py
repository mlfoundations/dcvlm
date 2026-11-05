import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="where to download the GRIT-20M parquets")
args = parser.parse_args()

snapshot_download(
    repo_id="zzliang/GRIT",
    repo_type="dataset",
    allow_patterns=["grit-20m/*snappy.parquet"],
    local_dir=args.output_dir,
)

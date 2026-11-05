from huggingface_hub import snapshot_download
import os

output_dir = "./MINT-1T-HTML"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+"/parquets", exist_ok=True)
os.makedirs(output_dir+"/shards", exist_ok=True)

snapshot_download(
    repo_id="mlfoundations/MINT-1T-HTML",
    repo_type="dataset",
    allow_patterns=["data_v1_1/*"],
    local_dir=os.path.join(output_dir, "parquets")
)
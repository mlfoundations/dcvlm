from huggingface_hub import snapshot_download
import os

output_dir = "./OmniCorpus-CC"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+"/parquets", exist_ok=True)
os.makedirs(output_dir+"/shards", exist_ok=True)

snapshot_download(
    repo_id="OpenGVLab/OmniCorpus-CC",
    repo_type="dataset",
    allow_patterns=["CC-MAIN-*/*"],
    local_dir=os.path.join(output_dir, "parquets")
)
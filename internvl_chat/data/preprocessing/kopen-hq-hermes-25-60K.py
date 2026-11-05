from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

snapshot_download(
    repo_id="MarkrAI/KOpen-HQ-Hermes-2.5-60K",
    repo_type="dataset",
    local_dir="kopen_hq_hermes_25_60K_data"
)

output_dir = "./kopen-hq-hermes-25-60K"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquets = [x for x in os.listdir("./kopen_hq_hermes_25_60K_data/data") if x.endswith(".parquet")]

for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./kopen_hq_hermes_25_60K_data/data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True, total=len(df)):

        input_ = row["input"]
        instruction_ = row["instruction"]
        output_ = row["output"]

        if input_ and len(input_) > 0:
            prompt_ = "{}\n\n{}".format(input_, instruction_)
        else:
            prompt_ = instruction_

        assert input_ is not None, "input_ is None!!!"
        assert instruction_ is not None, "instruction_ is None!!!"
        assert output_ is not None, "output_ is None!!!"

        convs = [{"from":"human", "value":"{}".format(prompt_)}, {"from":"gpt", "value":output_}]
        entry_ = {"id":len(new_jsonl_entries), "conversations": convs}
        new_jsonl_entries.append(entry_)

with open('./kopen-hq-hermes-25-60K/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('kopen_hq_hermes_25_60K_data')

print("Preprocessing complete! Saved as 'kopen-hq-hermes-25-60K/preprocessed.jsonl' with {} total samples.".format(len(new_jsonl_entries)))
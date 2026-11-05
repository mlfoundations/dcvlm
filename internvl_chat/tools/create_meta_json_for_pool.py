import os
import sys
import re
import json

pool_size = sys.argv[1]
assert pool_size in ["small", "medium", "large"], "invalid pool size - options include 'small', 'medium' and 'large'"

DATA_PATH = os.path.join(os.environ.get("DATA_ROOT", "/path/to/pools"), f"dmt_{pool_size}/")
META_PATH = f"./full_run_meta_{pool_size}_pool_LATEST.json"
CUMULATIVE_SUM_SAMPLES_FILE = "num_saved_samples.txt"
LOG_PATH = os.environ.get("LOG_DIR", "/path/to/logs")

DATASET_FORMAT_ROUTERS = {
    "relaion_2b": "_format_icpairs_from_img_txt",
    "grit_caption": "_format_icpairs_from_img_txt",
    "mint_html": "_format_mmdocs",
    "mint_pdf": "_format_mmdocs",
    "multimodal_textbook": "_format_mmdocs",
    "wanjuan": "_format_mmdocs",
    "asv1": "",  # NOTE: this is slightly different
    "omnicc": "_format_mmdocs",
    "datacomp1b": "_format_icpairs_from_img_txt",
    "recap1b": "",  # NOTE: this is slightly different
    "internvl_sa1b_caption": "_format_icpairs_from_img_txt_internvl_sa_1b",  # this is slightly weird, it has .txt keys instead of txt keys, hence added a separate function to avoid confusion.
    "facecaption": "_format_icpairs_from_img_txt",
}

def compute_length(dataset, pool_size, log_path):
    pattern = re.compile(r"Total samples processed:\s+(\d+)\s+into")
    matches = []
    with open(os.path.join(log_path, f"{dataset}_{pool_size}.out"), 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                matches.append(int(match.group(1)))

    if len(matches) == 0:
        raise ValueError("No matching line found in the file.")
    elif len(matches) > 1:
        raise ValueError("Multiple matching lines found in the file.")

    return matches[0]


metadict = {}

for dataset in os.listdir(DATA_PATH):

    dataset_path = os.path.join(DATA_PATH, dataset)
    tars = sorted(
            [f for f in os.listdir(dataset_path) if f.endswith(".tar")],
            key=lambda x: int(x.split('.')[0])
        )
    max_tar_idx = int(tars[-1].split('.')[0])
    root = f"{dataset_path}/{{000000..{max_tar_idx:06d}}}.tar"

    if os.path.exists(os.path.join(dataset_path, CUMULATIVE_SUM_SAMPLES_FILE)):
        with open(os.path.join(dataset_path, CUMULATIVE_SUM_SAMPLES_FILE), 'r') as f:
            length = int(f.read())
    else:
        print(f"Computing length from create_pool log files for dataset {dataset}...")
        length = compute_length(dataset, pool_size, LOG_PATH)
        with open(os.path.join(dataset_path, CUMULATIVE_SUM_SAMPLES_FILE), 'w') as f:
            f.write(str(length))
    curr_ = {
        "root": root,
        "data_augment": False,
        "repeat_time": 1,
        "length": length,
        "dataloading_class": "webdataset",
        "data_type": "instruction"
    }
 
    if dataset == "wanjuan_matteo":
        dataset = "wanjuan"

    if dataset.startswith("relaion"):
        dataset = "relaion-2b"
    elif dataset.startswith("omnicc"):
        dataset = "omnicc"
    elif dataset.startswith("mint_html"):
        dataset = "mint_html"
    if dataset in metadict:
        curr_["root"] += f"::{metadict[dataset]['root']}"
        curr_["length"] += metadict[dataset]['length']

    if dataset in list(DATASET_FORMAT_ROUTERS.keys()):
        curr_["data_type"] = "mmdoc_ic"
    if dataset.startswith("datacomp") or dataset.startswith("relaion") or dataset == "grit_caption" or dataset.startswith("internvl") or dataset.startswith("facecaption") or dataset == "as100m":
        curr_["data_type"] = "captioning"

    metadict[dataset] = curr_

with open(META_PATH, 'w') as f:
    json.dump(metadict, f, indent=4)
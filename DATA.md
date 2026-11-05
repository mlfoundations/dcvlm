## Setting Up Data

### Getting the data
- **Candidate pools and pre-mixed datasets.** The raw candidate pools for each competition scale, and the ready-to-train pre-mixed datasets, are on the 🤗 Hub; see the [Releases](README.md#-releases) section of the README.
- **Individual datasets.** `internvl_chat/data/preprocessing/` has one script per source dataset that downloads it and converts it to our format. The larger interleaved and image-text datasets (MINT-1T, OmniCorpus-CC, WanJuan, GRIT, SA-1B based datasets) have their own instructions in [`internvl_chat/data/preprocessing/interleaved/README.md`](internvl_chat/data/preprocessing/interleaved/README.md).

### Formatting
We follow the same data formatting as [InternVL chat format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html). Please carefully go over the InternVL docs to get a good understanding of data formatting for training. To convert individual annotation files from `json` to `jsonl` format, we provide a utility script: [`internvl_chat/tools/json2jsonl.py`](internvl_chat/tools/json2jsonl.py). To convert `jsonl` datasets into WebDataset shards, see [`converter.py`](converter.py).

### Meta JSON File

The trainer expects one meta json file (passed as `--meta_path`, or as `META_JSON` relative to `META_DIR` with the SLURM launchers) that lists every dataset in the mixture with its metadata. For example:

```json
{
    "chartqa": {
        "root": "/path/to/data/chartqa",
        "annotation": "/path/to/data/chartqa/preprocessed.jsonl",
        "data_augment": false,
        "repeat_time": 1,
        "length": 100000
    },
    "mint_html": {
        "root": "/path/to/data/mint_html/shards/{000000..000999}.tar",
        "data_augment": false,
        "repeat_time": 1,
        "length": 10000000,
        "dataloading_class": "webdataset",
        "data_type": "mmdoc_ic"
    }
}
```

`jsonl` datasets set `root` (the image folder) and `annotation`; WebDataset datasets set `root` to the shards and `"dataloading_class": "webdataset"`.

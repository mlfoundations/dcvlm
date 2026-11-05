import os
import sys
import logging
import webdataset as wds
import numpy as np
from pathlib import Path
from collections import OrderedDict
from webdataset.utils import pytorch_worker_info
from webdataset.shardlists import expand_urls


def log_and_continue_with_prefix(
    exn: Exception,
    prefix: str
) -> bool:
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'[{prefix}] WebDataset error ({repr(exn)}). Ignoring.')
    return True

def get_ds_name(url: str) -> str:
    return Path(url).parent.name

def update_sample(sample: dict, add_ds_name: bool = True, **kwargs) -> dict:
    """Perform dictionary update on a sample."""
    for k, v in kwargs.items():
        sample[k] = v
    if add_ds_name:
        sample["__ds_name__"] = get_ds_name(sample["__url__"])
    return sample

def identity(x):
    "Do nothing"
    return x

def worker_slice(src):
    """
    Return an iterator that slices the input iterator by worker id.
    This works both in single-node and multi-node setups.
    This has to be called inside a worker process.
    """
    rank, world_size, worker_id, num_workers = pytorch_worker_info(group=None)
    offset = worker_id + num_workers * rank
    splitsize = num_workers * world_size
    return wds.slice(offset, sys.maxsize, splitsize)(src)


def collate_dtype(
    meta: dict, 
    dtype: str,
    seed: int
):
    """
    Takes a Meta-Dictionary and collates all entries belonging to :dtype:
    into a single entry to be sliced by DataLoader workers. 

    Entries must have the same top-level "filters_meta_dir" or otherwise a ValueError will be thrown.
    """
    # separate :dtype: from the rest
    dtype_src_meta = {
        ds_name: ds_meta for ds_name, ds_meta in meta.items()
        if dtype == ds_meta["data_type"]
    }
    meta_nodtype = {
        ds_name: ds_meta for ds_name, ds_meta in meta.items()
        if dtype != ds_meta["data_type"]
    }
    if len(dtype_src_meta) == 0:
        raise ValueError(f"No meta entry for {dtype=}. Cannot collate.")
    
    # now we wanna preserve ordering across runs, so we first make sure our entries 
    # are sorted alphabetically, then we shuffle them based on a fixed seed
    dtype_keys = list(sorted(dtype_src_meta.keys()))
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(dtype_keys)
    dtype_src_meta = OrderedDict((key, dtype_src_meta[key]) for key in dtype_keys)

    # create a single big entry
    # TODO: support defaults
    heldout_meta =  {
        "root": [],
        "filters": {},
        "subject_to": {},
        "data_augment": False,
        "repeat_time": 1,
        "dataloading_class": "webdataset",
        "length": 0,
        "length_nested": {},
        "data_type": dtype,
        "filters_meta_dir": {} # made this a dict so we can stream small/medium/large altogether
    }
    empty_tars_full_paths = []
    for ds_name, ds_meta in dtype_src_meta.items():
        # necessary fields
        heldout_meta["root"].append(ds_meta["root"]) # concat all roots
        heldout_meta["length"] += float(ds_meta["length"]) # sum up all lengths
        heldout_meta["length_nested"][ds_name] = float(ds_meta["length"]) # used for capability mixing
        # filtering fields, which might not be there
        heldout_meta["filters"][ds_name] = ds_meta.get("filters", None) # nest all filters
        heldout_meta["subject_to"][ds_name] = ds_meta.get("subject_to", {}) # and all filter conditions
        heldout_meta["filters_meta_dir"][ds_name] = ds_meta.get("filters_meta_dir", None) # used for capability mixing
        # collect empty_tars as full paths to avoid basename collisions after collation
        for basename in ds_meta.get("empty_tars", []):
            root_dir = str(Path(ds_meta["root"]).parent) if not Path(ds_meta["root"]).is_dir() else ds_meta["root"]
            empty_tars_full_paths.append(os.path.join(root_dir, basename))
    
    # check if this is an unfiltered run
    has_filters = any(bool(value) for value in heldout_meta["filters"].values())
    if not has_filters:
        heldout_meta["filters"] = {}
        heldout_meta["subject_to"] = {}
        heldout_meta.pop("filters_meta_dir")
    else:
        # special kw so the :online_filtering: function knows to unpack the specs
        heldout_meta["filters"]["nested"] = True
    # brace-expand notation for multiple folders
    heldout_meta["root"] = "::".join(heldout_meta["root"])
    # carry over empty_tars as full paths so the pipeline can filter them
    if empty_tars_full_paths:
        heldout_meta["empty_tars"] = empty_tars_full_paths
    # add the concatenated entry back to the original Meta-Dictionary
    meta_nodtype.update({dtype: heldout_meta})
    return meta_nodtype


def total_workers(group=None):
    _, world_size, _, num_workers = pytorch_worker_info(group=group)
    return world_size*num_workers

def round_robin(src: list, upto: int):
    src_out = src.copy()
    while len(src_out) < upto:
        src_out.extend(src)
    return src_out[:upto]

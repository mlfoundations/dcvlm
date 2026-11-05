import os
import sys
import json
import logging
import webdataset as wds
from pathlib import Path
from functools import partial
from webdataset.shardlists import expand_urls
from webdataset.utils import pytorch_worker_info
from .formatters import (
    image_formatter, 
    conversation_formatter, 
    huggingface_formatter
)
from .mix import (
    DeterministicRandomMix, 
    WeightedResampledShards, 
    WeightedShardList
)
from .tariterators import tarfile_to_samples_with_metadata
from .filters import online_filter, decode_annotations
from .resume import StatefulUrls, StatefulSamples
from .utils import (
    update_sample, 
    identity, 
    worker_slice, 
    log_and_continue_with_prefix,
    collate_dtype,
    total_workers,
    round_robin
)

# done locally, per dataset
_SHARD_SHUFFLE_SIZE = 500
_SHARD_SHUFFLE_INITIAL = 100
# done globally, after mixing
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

def expand_urls_custom(path: str) -> list[str]:
    """
    Custom url expander based on WebDataset utility.
    Falls back to reading all tarfiles within a local folder
    if :path: is not a brace-expand string. Support non-lex sortable
    datasets like FineVision.
    """
    # first expand with wds utility
    urls = expand_urls(path)
    # if we get a list with one item, it's either a single tar
    # or a folder that couldn't be expanded (i.e., a non-braceexp string)
    if len(urls) == 1:
        # look for tars right within the folder with no recursion
        pathobj = Path(path)
        if pathobj.is_dir():
            urls = [str(tarfile) for tarfile in pathobj.glob("*.tar")]
    return urls

def build_byte_pipeline(
    meta: dict,
    ds_name: str,
    seed: int = None,
    allow_distributed_replacement: bool = False,
    stash_length_tokens: bool = False,
):

    # preliminary checks on meta dict
    if 'annotation' in meta:
        raise ValueError(
            f'{ds_name}:Key "annotation" exists. '
            'Are you sure this is a webdataset?'
        )
    if 'length' not in meta or meta['length'] <= 0: 
        raise ValueError(
            f'{ds_name}: We need length metadata for the webdataset. '
            'Currently, length either does not exist or is 0!'
        )

    logger = logging.getLogger()
    logger.info("Calling build_bytes_pipeline with the following parameters: %s", locals())

    # will use these to log truly sampled sources
    # and inject dataset-dependent behavior if needed
    data_type = meta["data_type"]
    key_adder = partial(
        update_sample,  
        __data_type__=data_type,
        __max_dynamic_patch__=meta['max_dynamic_patch'],
        __is_train__=meta['data_augment']
    )

    # define URL stream, which is either
    # - random replacement (is infinite by default)
    # - clear worker split (which we make infinite)
    if "empty_tars" in meta:
        all_urls = expand_urls_custom(meta['root'])
        empty_set = set(meta['empty_tars'])
        # match by full path if empty_tars contains paths (collated), else by basename (single dataset)
        has_paths = any('/' in t for t in empty_set)
        if has_paths:
            urls = [u for u in all_urls if u not in empty_set]
        else:
            urls = [u for u in all_urls if os.path.basename(u) not in empty_set]
        assert len(urls) > 0, f"All URLs are filtered out for dataset {ds_name}!"
        if len(urls) == len(all_urls):
            logger.warning(
                f"Meta Info for {ds_name} points to empty_tars, "
                "but found no matches when searching against URLs."
            )
        else:
            diff = len(all_urls) - len(urls)
            logger.warning(f"[{ds_name}] skipping {diff} empty tar(s) according to meta info.")
    else:
        urls = expand_urls_custom(meta['root']) # custom fn to handle parent dirs

    # special case, we're not allowing for repeats
    # but one entry has less URLs than workers; in such 
    # a case we need to replicate URLs round-robin across workers
    if (
        len(urls) < total_workers()
        and not allow_distributed_replacement
    ):
        raise ValueError(
            "You're passing --lazy_allow_distributed_replacement False, "
            f"but source `{ds_name}` only has {len(urls)} shards vs {total_workers()} total workers. "
            "This will inevitably cause within-epoch data repetition. "
            "Either reshard the data, or decrease the workers."
        )
    
    if seed is None:
        raise ValueError(
            "Seed cannot be None. Please set an explicit seed."
        )

    if allow_distributed_replacement:
        # each worker will randomly sample URLs 
        # with a different seed (worker-id+time+pid)
        if meta.get("length_nested", False):
            weights = meta["length_nested"] # dictionary of src: length
            url_streamer = WeightedResampledShards(
                urls,
                weights,
                seed=seed,
                deterministic=True
            )
        else:
            url_streamer = wds.ResampledShards(
                urls,
                seed=seed,
                deterministic=True
            )
        slicer = identity
        shard_shuffler = identity
    else:
        # uses the same, deterministic shard
        # ordering across workers, and shards are
        # sliced based on a flat notion of worker-id
        if meta.get("length_nested", False):
            weights = meta["length_nested"] # dictionary of src: length
            url_streamer = WeightedShardList(
                urls,
                weights,
                seed=seed,
                longest=True
            )
        else:
            url_streamer = wds.SimpleShardList(
                urls,
                seed=None # this won't shuffle the shards
            )
        slicer = worker_slice
        shard_shuffler = wds.detshuffle(
            bufsize=_SHARD_SHUFFLE_SIZE, 
            initial=_SHARD_SHUFFLE_INITIAL, 
            seed=seed
        )

    # define the data filter based on dataset specs
    all_filters = meta.get("filters", dict())
    if len(all_filters) == 0:
        logger.warning(
            f"No online filter specified for dataset {ds_name}. "
            "All samples will be kept."
        )
    filter_conditions = meta.get("subject_to", dict())
    filter_fn = partial(
        online_filter,
        online_filters=all_filters,
        filter_conditions=filter_conditions,
        verbose=False
    )

    # use custom tarfile_to_samples if annotations
    # metadata are given outside of the tarfiles
    if len(all_filters) > 0 and "filters_meta_dir" in meta:
        # logger.warning(
        #     f"Meta Info for {ds_name} points to using metadata-based filtering."
        #     f" Will read .json files from {meta['filters_meta_dir']}."
        # )
        tarfile_to_samples_fn = partial(
            tarfile_to_samples_with_metadata,
            filters_meta_dir=meta['filters_meta_dir']
        )
    else:
        tarfile_to_samples_fn = wds.tarfile_to_samples()

    # removes useless keys
    def cleanup(sample):
        keys_to_pop = []
        for k in sample.keys():
            # metadata added by img2dataset
            # or filters / external annotations
            if k.endswith("json"):  
                keys_to_pop.append(k)
            # stored by our image descriptor scripts, but not needed
            if k.endswith("phash.txt"): 
                keys_to_pop.append(k)
            if k.endswith("sha256.txt"):
                keys_to_pop.append(k)
            if k.endswith("sscd_disc_mixup.pt"):
                keys_to_pop.append(k)
        for k in set(keys_to_pop):
            sample.pop(k)
        return sample

    # stash the precomputed multimodal token length from the external/in-tar
    # annotations into a private key BEFORE `cleanup` drops all *json keys.
    # Used by the dataset-dump tooling to enforce a token budget identical to
    # training's; unused (and not inserted) during normal training.
    def stash_len(sample):
        annots = {}
        for annot_key in ('filters.json', 'metadata.json'):
            annots.update(decode_annotations(sample, annot_key))
        sample["__length_tokens__"] = annots.get("length_tokens", 0) or 0
        return sample

    # build and return the pipeline
    stages = [
        # yield shuffled urls
        # - if with replacement: random order regardless of worker
        # - if without replacement: deterministic order, sliced by worker
        url_streamer,
        shard_shuffler,
        # slice raw urls (before cpu ops)
        slicer,
        # resume url state (within worker)
        StatefulUrls(replacement=allow_distributed_replacement),
        # group samples
        tarfile_to_samples_fn,
        # resume sample state (within worker)
        StatefulSamples(),
        # augment sample dicts with info
        wds.map(key_adder),
        # filter out samples
        wds.select(filter_fn),
    ]
    # optionally preserve length_tokens before the json-dropping cleanup
    if stash_length_tokens:
        stages.append(wds.map(stash_len))
    # cleanup useless keys (saves decoding time)
    stages.append(wds.map(cleanup))
    pipeline = wds.DataPipeline(*stages)

    # make infinite
    if not allow_distributed_replacement:
        return pipeline.repeat(sys.maxsize)
    return pipeline


def build_all_pipelines(
    data_args,
    meta_path: str,
    allow_distributed_replacement: bool = True,
    seed: int = None,
    collate_dtypes: list = [],
    mmdoc_format: str = "mt",
    stash_length_tokens: bool = False
):
    logger = logging.getLogger()
    logger.info("Calling build_pipeline with the following parameters: %s", locals())

    # open the meta json and read datasets
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # tweak the meta-json to collate datasets from one (or more)
    # data-types into a single entry; this is bc many datasets might
    # have less shards than workers, hence lead to data repetition across ranks
    for coll_dtype in collate_dtypes:
        meta = collate_dtype(meta, dtype=coll_dtype, seed=seed)
    rank, _, wid, _ = pytorch_worker_info(group=None)
    if (rank == 0) and (wid == 0):
        logger.error(f"{json.dumps(meta, indent=4)}")

    # update formatting string for multimodal documents
    for ds_name in meta.keys():
        ds_meta = meta[ds_name]
        if ds_meta["data_type"] == "mmdoc":
            ds_meta["data_type"] = f"mmdoc_{mmdoc_format}"
        meta[ds_name] = ds_meta

    # build all bytes pipelines
    sources = []
    probs = []
    names = []
    metas = []
    for ds_name, ds_meta in meta.items():
        # optionally override num tiles for a dataset
        # NOTE: maybe worth for high-res datasets?
        if "max_dynamic_patch" not in ds_meta:
            ds_meta["max_dynamic_patch"] = data_args.max_dynamic_patch

        # build a pipeline that only yields dicts of bytes
        # this supports resuming from a saved state
        pipeline = build_byte_pipeline(
            meta=ds_meta,
            ds_name=ds_name,
            seed=seed,
            allow_distributed_replacement=allow_distributed_replacement,
            stash_length_tokens=stash_length_tokens,
        )

        # determine the sampling probability
        prob = ds_meta['length'] * ds_meta['repeat_time']
        sources.append(pipeline)
        probs.append(prob)
        names.append(ds_name)
        metas.append(ds_meta)

    return sources, probs, names, metas



def build_data_mix(
    sources,
    probs,
    names,
    metas,
    tokenizer,
    template_name: str,
    num_image_token: int,
    image_size: int,
    pad2square: bool,
    use_thumbnail: bool,
    dynamic_image_size: bool,
    min_dynamic_patch: int,
    normalize_type: str,
    max_images: int,
    max_tokens: int,
    use_packed_ds: bool = True,
    allow_distributed_replacement: bool = True,
    seed: int = None
):
    logger = logging.getLogger()
    logger.info("Calling build_pipeline with the following parameters: %s", locals())

    # normalize probs
    probs = [p / sum(probs) for p in probs]

    # mix through the pipelines
    # edit: using a custom class for deterministic support
    mix = DeterministicRandomMix(
        sources, 
        seed=seed, 
        probs=probs, 
        longest=False
    )
    
    # define the image formatter
    im_formatter = partial(
        image_formatter,
        pad2square=pad2square,
        normalize_type=normalize_type,
        num_image_token=num_image_token,
        dynamic_image_size=dynamic_image_size,
        min_dynamic_patch=min_dynamic_patch,
        max_images=max_images,
        image_size=image_size,
        use_thumbnail=use_thumbnail,
    )

    # define the conversation formatter
    conv_formatter = conversation_formatter

    # define the hf formatter
    hf_formatter = partial(
        huggingface_formatter,
        template_name=template_name,
        tokenizer=tokenizer,
        use_packed_ds=use_packed_ds,
        max_tokens=max_tokens,
        persist_private_keys=True
    )

    # define all handlers
    decode_handler = partial(
        log_and_continue_with_prefix,
        prefix=f"decode_fn"
    )
    im_handler = partial(
        log_and_continue_with_prefix, 
        prefix=f"im_formatter"
    )
    conv_handler = partial(
        log_and_continue_with_prefix, 
        prefix=f"conv_formatter"
    )
    hf_handler = partial(
        log_and_continue_with_prefix, 
        prefix=f"hf_formatter"
    )

    # removes useless keys
    def cleanup(sample):
        sample.pop("__url__")
        sample.pop("__data_type__")
        sample.pop("__max_dynamic_patch__")
        sample.pop("__is_train__")
        return sample

    # build and return the pipeline
    return wds.DataPipeline(
        # yield sample bytes with weighting
        mix,
        # shuffle samples 
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
            seed=seed
        ), 
        # decode everything (images and UTF-8)
        wds.decode("pilrgb", handler=decode_handler), 
        # dynamic preprocessing for images
        wds.map(im_formatter, handler=im_handler),
        # ChatML format
        # NOTE: (this may change if we do native mm pt) 
        wds.map(conv_formatter, handler=conv_handler),
        # yields samples to be consumed by the HF-trainer
        wds.map(hf_formatter, handler=hf_handler),  
        wds.map(cleanup)
    )

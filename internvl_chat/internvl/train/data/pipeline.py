import os
import json
import logging
import webdataset as wds
from functools import partial
from .formatters import (
    image_formatter, 
    conversation_formatter, 
    huggingface_formatter
)
from .filters import online_filter
from .utils import (
    fix_prob, 
    update_sample, 
    identity, 
    worker_slice, 
    log_and_continue_with_prefix
)


_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

def worker_seed(seed: int) -> int:
    return seed


def build_byte_pipeline(
    meta: dict,
    ds_name: str,
    allow_distributed_replacement: bool = True,
    seed: int = None
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

    # grab attributes
    urls = meta['root']
    data_type = meta.get("data_type", "captioning")

    # will use these to log truly sampled sources
    # and inject dataset-dependent behavior if needed
    key_adder = partial(
        update_sample, 
        __ds_name__=ds_name, 
        __data_type__=data_type,
        __max_dynamic_patch__=meta['max_dynamic_patch'],
        __is_train__=meta['data_augment']
    )

    # define resampled shards to either
    # accept or reject distributed replacement
    if allow_distributed_replacement:
        # uses pytorch wid + time + pid for seeding
        # so shard order is different across workers
        url_streamer = wds.ResampledShards(
            urls,
            worker_seed=None, 
            deterministic=False
        )
        slicer = identity
    else:
        # uses the same, deterministic shard
        # ordering across workers, and shards are
        # sliced based on a flat notion of worker-id
        if seed is None:
            raise ValueError(
                f"Seed cannot be None when allow_distributed_replacement is False."
                " Please set an explicit seed."
            )
        url_streamer = wds.SimpleShardList(urls, seed=seed)
        slicer = worker_slice

    # define the data filter based on dataset specs
    filter_spec = meta.get("filters", dict())
    if len(filter_spec) == 0:
        logger.warning(
            f"No online filter specified for dataset {ds_name}. "
            "All samples will be kept."
        )
    filter_fn = partial(
        online_filter,
        online_filters=filter_spec
    )

    # build and return the pipeline
    return wds.DataPipeline(
        # - infinite sampling *with* replacement
        # - or finite sampling *without* replacement
        url_streamer,
        # slice raw urls (before cpu ops)
        slicer,
        # group samples
        wds.tarfile_to_samples(),
        # augment sample dicts with info
        wds.map(key_adder),
        # filter out samples
        wds.select(filter_fn),
    )


def build_mixed_pipeline(
    data_args,
    meta_path: str,
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

    # open the meta json and read datasets
    with open(meta_path, 'r') as f:
        meta = json.load(f)

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
        pipeline = build_byte_pipeline(
            meta=ds_meta,
            ds_name=ds_name,
            allow_distributed_replacement=allow_distributed_replacement,
            seed=seed
        )

        # determine the sampling probability
        prob = ds_meta['length'] * ds_meta['repeat_time']
        sources.append(pipeline)
        probs.append(prob)
        names.append(ds_name)
        metas.append(ds_meta)

    # normalize probs
    probs = [p / sum(probs) for p in probs]
    # fix the probs in case distributed replacement is off
    if not allow_distributed_replacement:
        for i, (name, ds_meta, prob) in enumerate(zip(names, metas, probs)):
            prob = fix_prob(name, ds_meta['root'], prob)
            probs[i] = prob
        probs = [p / sum(probs) for p in probs]

    # mix through the pipelines
    mix = wds.RandomMix(sources, probs=probs, longest=False)
    
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
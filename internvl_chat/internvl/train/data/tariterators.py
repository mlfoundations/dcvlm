import os
import gc
import json
import logging
from time import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional
from webdataset import gopen
from webdataset.tariterators import tar_file_iterator, group_by_keys, base_plus_ext
from webdataset.handlers import reraise_exception
from .utils import get_ds_name

logger = logging.getLogger()

def log_and_continue(exn: Exception) -> bool:   
    """Log exception and continue."""
    print(f"Exception {repr(exn)} in log_and_continue")
    return True

class FiltersMetadata:
    """
    Metadata for filters.
    {original_filename}.json:
    "{original_key}": {
        "Filter_1": value,
        "Filter_2": value,
        …
    }
    meta.json
    "filters": {
            "filters_meta_dir": "dmt_filtered_iqa_test",
            "filters": {
                "random_score": {
                    "threshold": 0.5,
                    "sign": ">"
                }
            }
        }
    """
    def __init__(self, source):
        json_data = {}
        if "filters_meta_stream" in source:
            json_fileobj = source["filters_meta_stream"]
            json_fileobj_bytes = json_fileobj.read()
            json_data = json.loads(json_fileobj_bytes)
            assert isinstance(json_data, dict)
        self.json_data = json_data

    def __getitem__(self, key):
        if key in self.json_data:
            fname = f"{key}.metadata.json"
            return {
                    "fname": fname,
                    "data": json.dumps(self.json_data[key])
                }

def custom_tar_file_expander(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    keys: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    eof_value: Optional[Any] = {},
) -> Iterator[Dict[str, Any]]:
    """Expand tar files.

    Args:
        data: Iterator over opened tar file streams.
        handler: Exception handler.
        select_files: Select files from tarfiles by name (permits skipping files).
        rename_files: Function to rename files.
        eof_value: Value to yield at the end of each shard.

    Yields:
        A stream of samples.
    """
    for source in data:
        url = source["url"]
        local_path = source.get("local_path")

        try:
            assert isinstance(source, dict)
            assert "stream" in source
            assert "filters_meta_stream" in source
            assert "meta_url" in source
            meta_url = source["meta_url"]

            filters_metadata = FiltersMetadata(source)
            sample_keys = set()
            for sample in tar_file_iterator(
                    source["stream"],
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                ):
                # these lines are taken from wds, no changes
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                sample["__url__"] = url
                fname = sample["fname"]
                if local_path is not None:
                    sample["__local_path__"] = local_path

                # NOTE: here the changes start
                # the first time we hit a new key,
                # we yield the corresponding metadata together with
                # the current sample (i.e., the chunk of the tarfile, not the finalized dict)
                sample["__meta_url__"] = meta_url
                prefix, _ = keys(fname)
                if prefix is None:
                    continue
                if prefix not in sample_keys:
                    # we grab the corresponding metadata
                    sample_keys.add(prefix)
                    filters_sample = filters_metadata[prefix]
                    
                    # if we find metadata, we yield it
                    if filters_sample is not None:
                        filters_sample["__url__"] = url
                        if local_path is not None:
                            filters_sample["__local_path__"] = local_path
                        for s_ in [filters_sample, sample]:
                            yield s_
                else:
                    yield sample

            # we yield an EOF marker at the end of each shard so that
            # samples from different shards don't get mixed up
            if eof_value is not None:
                yield eof_value

            # release memory at the end of each shard
            del filters_metadata
            del sample_keys
            gc.collect()
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def custom_url_opener(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    **kw: Dict[str, Any],
):
    """(Edited from webdataset)
    Open URLs and yield a stream of url+stream pairs.

    Args:
        data: Iterator over dict(url=...).
        handler: Exception handler.
        **kw: Keyword arguments for gopen.gopen.

    Yields:
        A stream of url+stream pairs.
    """
    # pop from kw so that we don't pass it to gopen
    filters_meta_dir = kw.pop("filters_meta_dir", None)
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            if filters_meta_dir is not None:
                if isinstance(filters_meta_dir, dict):
                    ds_name = get_ds_name(url)
                    meta_dir_for_url = filters_meta_dir.get(ds_name, None)
                    if meta_dir_for_url is None:
                        raise ValueError(
                            f"Url={url} ({ds_name=}) doesn't have a corresponding entry in the filters_meta_dictionary."
                            "Cannot load annotations for this tarfile."
                        )
                elif isinstance(filters_meta_dir, str):
                    meta_dir_for_url = filters_meta_dir
                # try to search at a basename-level
                # e.g. /root/dataset/000.tar -> /annots/dataset/000.json
                file_basename = Path(url).with_suffix(".json").name 
                filters_meta_path = Path(meta_dir_for_url) / file_basename
                if not filters_meta_path.exists():
                    msg = f"{filters_meta_path} doesn't exist"
                    # otherwise try to search from the top-level
                    data_root = Path(url).parent.parent
                    file_basename = Path(url).relative_to(data_root).with_suffix(".json")
                    filters_meta_path = Path(meta_dir_for_url) / file_basename
                    msg += f" looking for {filters_meta_path}"
                    logger.error(msg)
                    if not filters_meta_path.exists():
                        raise ValueError(
                            f"URL {url} doesn't have a corresponding annotation file. "
                            f"MetaDir is set to {meta_dir_for_url}"
                        )
                filters_meta_path = str(filters_meta_path)
                json_stream = gopen(filters_meta_path, **kw)
                # update fields in the sample
                sample["filters_meta_dir"] = meta_dir_for_url
                sample.update(filters_meta_stream=json_stream, meta_url=filters_meta_path)
            stream = gopen(url, **kw)
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def tarfile_to_samples_with_metadata(src, filters_meta_dir, keys=base_plus_ext, handler=reraise_exception):
    streams = custom_url_opener(src, handler=handler, filters_meta_dir=filters_meta_dir)
    files = custom_tar_file_expander(streams, keys=keys, handler=handler)
    samples = group_by_keys(files, keys=keys, handler=handler)
    return samples
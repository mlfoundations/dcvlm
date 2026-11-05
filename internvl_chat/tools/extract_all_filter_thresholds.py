import os
import json
import time
import sys
import torch
from glob import glob
from math import ceil, floor
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import webdataset as wds
from webdataset.tariterators import url_opener, tar_file_expander, group_by_keys

# file to save thresholds corresponding to selection based on different filters with different percentiles
THRESHOLD_METADATA = 'extracted_percentiles.json'
PERCENTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def log_and_continue(exn):
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def decode_filter_json(sample):
    for k in sample.keys():
        if 'filters.json' in k:
            sample[k] = json.loads(sample[k])
    return sample


def tarfile_to_samples_throw_wrapped(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys(files, handler=handler)
    return samples


def collate_(batch):    
    return batch


def torch_quantile(tensor, q, dim=None, interpolation="linear"):
    """The code below is taken directly from https://github.com/pytorch/pytorch/issues/157431#issuecomment-3026856373.
    
    Improved ``torch.quantile`` for one scalar quantile.

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.

    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:

    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.
    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: intepolation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    return out.squeeze() if dim_was_none else out.squeeze(dim)


def main(args):
    # check if the threshold metadata file already exists, if so load processed filters
    # expected output: dict where key = filter, value = {0.1: some threshold value, ..., 0.9: some threshold value}
    output_path = os.path.join(args.input_folder, THRESHOLD_METADATA)
    if os.path.exists(output_path):
        with open(output_path, 'r') as fin:
            all_filter_percentiles = json.load(fin)
        existing_filters = list(all_filter_percentiles.keys())
    else:
        all_filter_percentiles = defaultdict(dict)
        existing_filters = []

    # flat list of tarfiles in the current directory
    tarfiles = sorted(glob(os.path.join(args.input_folder, "*.tar"), recursive=False))

    # get brace-expanded format for wds
    brace_expanded = os.path.join(args.input_folder, '{' + '{}..{}'.format(os.path.basename(tarfiles[0]).removesuffix(".tar"), os.path.basename(tarfiles[-1]).removesuffix(".tar")) + '}.tar')

    # get wds pipeline
    decoded_dataset = wds.DataPipeline(
        wds.SimpleShardList(brace_expanded),  # this will give an iterator over all tars
        wds.split_by_worker,  # this will distribute across all cpu workers within a rank (num_workers per gpu)
        tarfile_to_samples_throw_wrapped,  # this will give us sample dicts per worker
        wds.map(decode_filter_json, handler=log_and_continue),
        wds.batched(args.batch_size, collation_fn=collate_),
    )

    # create dataloader for batched dataloading
    dataloader = wds.WebLoader(
        decoded_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # get time stats
    st = time.time()
    # gather the scores from all samples for all filters
    # save them in a dict where key is filter name, value is list of N_SAMPLES scores corresponding to that filter
    new_percentiles = defaultdict(list)

    for index, batch in enumerate(tqdm(dataloader)):
        for dec_sample in batch:
            filter_data = dec_sample["filters.json"]
            for filter, value in filter_data.items():
                # if filter is already processed and corresponding thresholds are already extracted
                if filter in existing_filters:
                    continue
                # filter output can be None when a filter could not be run on a sample for whatever reason
                if value is None:
                    value = 0
                # convert to float32 (from float64) to save memory
                value_float32 = torch.tensor([value], dtype=torch.float32)
                new_percentiles[filter].append(value_float32)
    
    for filter, value_tensor in new_percentiles.items():
        # flatten the list of float tensors
        value_tensor = torch.stack(value_tensor, dim=1).squeeze(0)

        # torch.quantile() only works for arrays with fewer than 16M samples 
        # So for large datasets we use a customized implementation, torch_quantile(), that processes a specific quantile at a time
        # NOTE: check this for each pool size. The datasets that exceed 16M samples may vary with the pool size. 
        if 'datacomp_1b' or 'relaion-2b-en-vol-1-unpacked-TEST' in args.input_folder:
            thres = []
            for q in PERCENTILES:
                thres.append(torch_quantile(value_tensor, q, interpolation=args.interpolation).item())
                print(q, thres, time.time() - st)

        else:
            thres = [float(t) for t in (torch.quantile(value_tensor, torch.tensor(PERCENTILES), interpolation=args.interpolation).numpy())]

        # update the output threshold dict with new filters and their thresholds
        all_filter_percentiles[filter] = {PERCENTILES[idx]: thres[idx] for idx in range(len(PERCENTILES))}

    with open(output_path, 'w') as fout:
        json.dump(all_filter_percentiles, fout, indent=4)
    print("Took {} seconds to compute percentiles for all filters!".format(time.time()-st))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--interpolation', type=str, default='linear')
    args = parser.parse_args()

    main(args)

import os
import sys
import time
import random
import numpy as np
import webdataset.utils as wds_utils
from torch.utils.data import IterableDataset
from collections import defaultdict
from .utils import get_ds_name


class DeterministicRandomMix(IterableDataset):
    """Iterate over multiple datasets by randomly selecting samples based on given probabilities."""

    def __init__(self, datasets, seed, worker_seed=None, probs=None, longest=False):
        """Initialize the RandomMix iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            probs (list, optional): List of probabilities for each dataset. Defaults to None.
            longest (bool): If True, continue until all datasets are exhausted. Defaults to False.
        """
        self.datasets = datasets
        self.probs = probs
        self.longest = longest
        self.seed = seed
        self.worker_seed = wds_utils.pytorch_worker_seed if worker_seed is None else worker_seed
        # fixed generator
        self.seed = wds_utils.make_seed(self.worker_seed(), 0, self.seed)
        self.rng = np.random.default_rng(self.seed)
    
    def get_state(self):
        return self.rng.bit_generator.state
    
    def set_state(self, state):
        if state is not None:
            self.rng.bit_generator.state = state

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        # iterators to pick from
        sources = [iter(d) for d in self.datasets]
    
        # define sampling ratios
        if self.probs is None:
            probs = [1] * len(sources)
        else:
            probs = list(self.probs) # make a copy
        
        # start yielding
        while len(sources) > 0:
            norm_probs = np.asarray(probs)
            norm_probs /= norm_probs.sum()
            i = self.rng.choice(len(norm_probs), p=norm_probs)
            try:
                yield next(sources[i])
            except StopIteration:
                if self.longest:
                    del sources[i]
                    del probs[i]
                else:
                    break


class WeightedResampledShards(IterableDataset):
    """An iterable dataset yielding a list of URLs with resampling."""

    def __init__(
        self,
        urls,
        weights: dict,
        nshards=sys.maxsize,
        seed=0,
        worker_seed=None,
        deterministic=False,
        empty_check=True,
    ):
        """Very similar to WebDataset's ResampledShards, but 
        shards may have different sampling probabilities.
        """
        super().__init__()
        self.urls = urls
        self.weights = weights
        if empty_check:
            if len(self.urls) == 0:
                raise ValueError("empty_check=True, but no shards found in WeightedResampledShards")
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = wds_utils.pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.seed = seed
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        # no changes wrt to ResampledShards here
        self.epoch += 1
        if self.deterministic:
            seed = wds_utils.make_seed(self.worker_seed(), self.epoch, self.seed)
        else:
            seed = wds_utils.make_seed(
                self.worker_seed(),
                self.epoch,
                self.seed,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")
        self.rng = np.random.default_rng(seed)
        
        # changes start from here
        # 1 - init sources and corresponding probs
        source_names, probs = [], []
        for src_name, weight in self.weights.items():
            source_names.append(src_name)
            probs.append(weight)
        probs = np.array(probs)
        probs /= probs.sum()

        # 2 - init list of URLs belonging to src
        src2urls = defaultdict(list)
        for url in self.urls:
            src_name = get_ds_name(url)
            assert src_name in source_names
            src2urls[src_name].append(url)

        for _ in range(self.nshards):
            # pick a source according to probs
            src_idx = self.rng.choice(len(source_names), p=probs)
            src = source_names[src_idx]
            # pick a shard from src
            shard = self.rng.choice(src2urls[src]) # uniform
            yield dict(url=shard)



class WeightedShardList(IterableDataset):
    def __init__(
        self,
        urls,
        weights: dict,
        seed: int,
        longest: bool = False
    ):
        """Initialize a ShardList where shards may have weights
        according to their data source specified. The get_ds_name()
        function governs url-to-source assignment, and each source must
        have a corresponding weight in the :weights: argument.

        Args:
            urls: A list of URLs.
            weights (dict): Sampling weights for each independent source.
            seed (int): The seed for random number generation.
            longest (bool): If True, do not terminate after one source is exhausted.
        """
        super().__init__()
        self.urls = urls
        self.weights = weights
        assert isinstance(self.urls[0], str) and isinstance(self.urls, list)
        assert seed is not None
        self.seed = seed
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        import logging
        logger = logging.getLogger()
        
        # changes start from here
        # 1 - init sources and corresponding probs
        source_names, probs = [], []
        for src_name, weight in self.weights.items():
            source_names.append(src_name)
            probs.append(weight)
        probs = np.array(probs)
        probs /= probs.sum()

        # 2 - init list of URLs belonging to each src
        src2urls = defaultdict(list)
        for url in self.urls:
            src_name = get_ds_name(url)
            assert src_name in source_names
            src2urls[src_name].append(url)

        # init rng
        self.rng = np.random.default_rng(self.seed)

        # start src-weighted yielding 
        while len(src2urls) > 0:
            # pick a source according to probs
            src_idx = self.rng.choice(len(source_names), p=probs)
            src = source_names[src_idx]
            # check if we hit an empty source
            if len(src2urls[src]) == 0:
                # keep iterating if a source is exhausted
                if self.longest:
                    del source_names[src_idx]
                    probs = np.delete(probs, src_idx)
                    probs /= probs.sum()
                    del src2urls[src]
                    continue
                else:
                    break
            # pop a URL from src
            yield dict(url=src2urls[src].pop(0))
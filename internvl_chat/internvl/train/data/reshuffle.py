"""
Shared helpers for the two-level global reshuffle of a finalized DCVLM-Baseline
release (see reshuffle_scatter.py / reshuffle_gather.py).

WHY A RESHUFFLE (measured, not assumed)
---------------------------------------
The dump writes each 10k-sample shard as one worker's live draw-order window.
The production mix + `wds.shuffle(5000)` decorrelate FINE-GRAINED ORDER (measured
run-length of consecutive same-source samples ~= 1.13) but NOT shard-scale SOURCE
COMPOSITION: a single shard is drawn from whatever handful of input shards the
workers had open, so its per-source rates are far from global (e.g. one shard had
figureqa at 16.7% vs 0.87% global = 19x; only 26/166 sources present). The design
effect across sources is Deff ~= 633 (median). Online training is unaffected
because it aggregates across ~2048 rank x workers every optimizer step and covers
the full epoch; but an external user who streams a SUBSET of shards, or uses a
normal (few-thousand) shuffle buffer, inherits the per-shard skew. This reshuffle
bakes the cross-worker averaging into the on-disk shard layout so every output
shard is independently representative (Deff -> ~1), matching MAmmoTH-VL.

MECHANISM (two-level MapReduce, external shuffle)
-------------------------------------------------
A one-pass "assign each sample a random FINAL shard" needs ~N_shards (10k+) open
writers per task -> FD blow-up. So we split it:

  Pass 1 (scatter, reshuffle_scatter.py): every task reads a disjoint slice of the
    input shards and routes each sample (raw bytes, NO decode) to one of B buckets
    chosen UNIFORMLY AT RANDOM, independent of the sample's source. Because the
    routing is source-independent, each bucket receives ~1/B of EVERY source ->
    each bucket is already a representative random subsample of the whole corpus.
    B (~128-256) is small enough to keep all bucket writers open at once.

  Pass 2 (gather, reshuffle_gather.py): each task reads a whole bucket, applies a
    large in-RAM `wds.shuffle` (to break the residual within-part ordering left by
    each pass-1 task's input-shard walk), and writes final shards. Since bucket
    membership is already representative, chunking a shuffled bucket into 10k-sample
    shards yields representative shards.

Everything is lossless raw-byte pass-through (`tarfile_to_samples` -> ShardWriter);
samples keep their globally-unique `__key__`, so read-back is byte-identical.
"""

import os
import glob

import numpy as np
import webdataset as wds
import webdataset.utils as wds_utils


# scatter/gather directory layout under the reshuffle root
SCATTER_DIRNAME = "scatter"          # scatter/bucket_XXXX/tYYYYY-%06d.tar
SCATTER_MARKERS = "scatter_markers"  # scatter_markers/tYYYYY.done
GATHER_DIRNAME = "shuffled"          # shuffled/bXXXX-%06d.tar
GATHER_MARKERS = "gather_markers"    # gather_markers/bXXXX.done


def bucket_dir(root: str, b: int) -> str:
    return os.path.join(root, SCATTER_DIRNAME, f"bucket_{b:04d}")


def list_input_shards(release_dir: str) -> list:
    """Sorted list of finalized input shards (*.tar) to reshuffle."""
    shards = sorted(glob.glob(os.path.join(release_dir, "*.tar")))
    if not shards:
        raise SystemExit(f"No *.tar shards found in {release_dir}")
    return shards


def raw_sample_stream(shard_paths):
    """Yield grouped samples as raw-byte dicts (no image/text decoding).

    Each yielded dict has `__key__` plus one entry per member (field -> bytes),
    exactly what `wds.ShardWriter.write` re-serializes byte-for-byte.
    """
    pipe = wds.DataPipeline(
        wds.SimpleShardList(list(shard_paths)),
        wds.tarfile_to_samples(),
    )
    return iter(pipe)


def writable(sample: dict) -> dict:
    """Strip webdataset-internal keys that ShardWriter cannot serialize."""
    return {k: v for k, v in sample.items() if not (k.startswith("__") and k != "__key__")}


def rank_seeded_rng(seed: int, rank: int) -> np.random.Generator:
    """Deterministic per-rank RNG so pass-1 routing is reproducible (=> resumable:
    a re-run of the same rank reproduces the identical bucket assignments)."""
    return np.random.default_rng(wds_utils.make_seed(seed, rank))

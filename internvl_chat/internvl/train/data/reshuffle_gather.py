"""
Pass 2 of the global reshuffle: GATHER (per-rank CLI).

Each rank owns the buckets `b` with `b % WORLD_SIZE == RANK`. For each owned
bucket it reads every scattered part (`scatter/bucket_<b>/*.tar`), applies a large
in-RAM `wds.shuffle` to break the residual within-part ordering, and writes final
shards `<root>/shuffled/b<b>-%06d.tar`. Because pass-1 routing was source-
independent, each bucket is already a representative subsample; the shuffle only
randomizes order, so each 10k-sample output shard is representative (Deff -> ~1).

Idempotent per bucket: a bucket with `<root>/gather_markers/b<b>.done` is skipped.
To restart a bucket, delete its `shuffled/b<b>-*.tar` first.

Run (one task per rank, via srun):
  RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS \
  python -m internvl.train.data.reshuffle_gather \
    --root $OUT/reshuffle --num_buckets 256 \
    --samples_per_shard 10000 --shuffle_bufsize 100000 --seed 40
"""

import os
import glob
import json
import time
import random
import argparse

import webdataset as wds

from .reshuffle import GATHER_DIRNAME, GATHER_MARKERS, bucket_dir, writable


def gather_one_bucket(args, b: int, out_dir: str, markers_dir: str):
    done_marker = os.path.join(markers_dir, f"b{b:04d}.done")
    if os.path.exists(done_marker):
        print(f"[gather] bucket {b}: marker exists, skipping.", flush=True)
        return
    parts = sorted(glob.glob(os.path.join(bucket_dir(args.root, b), "*.tar")))
    if not parts:
        print(f"[gather] bucket {b}: no parts, writing empty marker.", flush=True)
        payload = {"bucket": b, "n_gathered": 0, "n_parts": 0,
                   "shuffle_bufsize": args.shuffle_bufsize, "seconds": 0.0}
        tmp = done_marker + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, done_marker)
        return
    # Randomize the PART read order so a bounded buffer mixes many scatter-ranks
    # (each part is one scatter-rank's samples routed here, in that rank's read
    # order; reading parts in sorted order would concatenate rank-correlated runs).
    random.Random(args.seed + b).shuffle(parts)
    # STREAM through a bounded shuffle buffer: initial < bufsize guarantees the
    # pipeline yields (and ShardWriter flushes) while reading, so RAM is capped at
    # ~bufsize samples (~230KB each) instead of loading the whole bucket. Bucket
    # membership is already representative (source-independent scatter), so a modest
    # buffer over randomized parts is enough to decorrelate the output shards.
    initial = min(args.shuffle_bufsize, max(1, args.shuffle_bufsize // 2))
    pipe = wds.DataPipeline(
        wds.SimpleShardList(parts),
        wds.tarfile_to_samples(),
        wds.shuffle(bufsize=args.shuffle_bufsize, initial=initial, seed=args.seed + b),
    )
    sink = os.path.join(out_dir, f"b{b:04d}-%06d.tar")
    t0 = time.time()
    n = 0
    with wds.ShardWriter(sink, maxcount=args.samples_per_shard, maxsize=10e10, verbose=0) as w:
        for sample in pipe:
            w.write(writable(sample))
            n += 1
    payload = {"bucket": b, "n_gathered": n, "n_parts": len(parts),
               "shuffle_bufsize": args.shuffle_bufsize, "seconds": time.time() - t0}
    tmp = done_marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, done_marker)
    print(f"[gather] bucket {b}: DONE — {n:,} samples from {len(parts)} parts "
          f"in {time.time()-t0:.0f}s", flush=True)


def main(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    out_dir = os.path.join(args.root, GATHER_DIRNAME)
    markers_dir = os.path.join(args.root, GATHER_MARKERS)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)

    my_buckets = [b for b in range(args.num_buckets) if b % world_size == rank]
    print(f"[gather] rank {rank}/{world_size}: owns {len(my_buckets)} buckets {my_buckets[:8]}"
          f"{' ...' if len(my_buckets) > 8 else ''}", flush=True)
    for b in my_buckets:
        gather_one_bucket(args, b, out_dir, markers_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Global reshuffle pass 2 (gather + shuffle each bucket).")
    p.add_argument("--root", required=True, help="Reshuffle work root (holds scatter/ from pass 1).")
    p.add_argument("--num_buckets", type=int, default=256)
    p.add_argument("--samples_per_shard", type=int, default=10_000)
    p.add_argument("--shuffle_bufsize", type=int, default=20_000,
                   help="Bounded streaming shuffle buffer (~230KB/sample => 20k ~= 4.6GB RAM/rank). "
                        "Combined with randomized part order; bucket membership is already "
                        "representative so this need not span the whole bucket.")
    p.add_argument("--seed", type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

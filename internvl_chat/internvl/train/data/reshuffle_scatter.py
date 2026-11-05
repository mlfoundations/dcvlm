"""
Pass 1 of the global reshuffle: SCATTER (per-rank CLI).

Each rank reads a disjoint, strided slice of the finalized input shards and routes
every sample (raw bytes, no decode) to one of `--num_buckets` buckets chosen
uniformly at random. Output goes to `<root>/scatter/bucket_<b>/r<rank>-%06d.tar`;
rank-prefixed names mean ranks never collide writing into the same bucket dir.

Idempotent: a rank with `<root>/scatter_markers/r<rank>.done` is skipped. To
restart an unfinished rank, delete its `scatter/bucket_*/r<rank>-*.tar` parts
first (routing is deterministic in (seed, rank), so the re-run reproduces it).

Run (one task per rank, via srun):
  RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS \
  python -m internvl.train.data.reshuffle_scatter \
    --release_dir $OUT/release --root $OUT/reshuffle \
    --num_buckets 256 --samples_per_shard 10000 --seed 40
"""

import os
import json
import time
import argparse

import webdataset as wds

from .reshuffle import (
    SCATTER_MARKERS, bucket_dir, list_input_shards,
    raw_sample_stream, writable, rank_seeded_rng,
)


def main(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    markers_dir = os.path.join(args.root, SCATTER_MARKERS)
    os.makedirs(markers_dir, exist_ok=True)
    done_marker = os.path.join(markers_dir, f"r{rank:05d}.done")
    if os.path.exists(done_marker):
        print(f"[scatter] rank {rank}/{world_size}: marker exists, skipping.", flush=True)
        return

    B = args.num_buckets
    for b in range(B):
        os.makedirs(bucket_dir(args.root, b), exist_ok=True)

    all_shards = list_input_shards(args.release_dir)
    my_shards = all_shards[rank::world_size]      # strided disjoint slice
    rng = rank_seeded_rng(args.seed, rank)

    # one open ShardWriter per bucket (B <= a few hundred << ulimit -n)
    writers = [
        wds.ShardWriter(
            os.path.join(bucket_dir(args.root, b), f"r{rank:05d}-%06d.tar"),
            maxcount=args.samples_per_shard, maxsize=10e10, verbose=0,
        )
        for b in range(B)
    ]

    t0 = time.time()
    n = 0
    per_bucket = [0] * B
    print(f"[scatter] rank {rank}/{world_size}: {len(my_shards)} input shards -> {B} buckets",
          flush=True)
    try:
        for sample in raw_sample_stream(my_shards):
            b = int(rng.integers(B))
            writers[b].write(writable(sample))
            per_bucket[b] += 1
            n += 1
            if n % (50 * args.samples_per_shard) == 0:
                print(f"[scatter] rank {rank}: {n:,} samples scattered "
                      f"({(time.time()-t0):.0f}s)", flush=True)
    finally:
        for w in writers:
            w.close()

    payload = {
        "rank": rank, "world_size": world_size, "num_buckets": B,
        "n_scattered": n, "per_bucket": per_bucket, "seed": args.seed,
        "n_input_shards": len(my_shards), "seconds": time.time() - t0,
    }
    tmp = done_marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, done_marker)
    print(f"[scatter] rank {rank}: DONE — {n:,} samples in {time.time()-t0:.0f}s", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Global reshuffle pass 1 (scatter to random buckets).")
    p.add_argument("--release_dir", required=True, help="Finalized input release dir (holds *.tar).")
    p.add_argument("--root", required=True, help="Reshuffle work root (holds scatter/ + markers).")
    p.add_argument("--num_buckets", type=int, default=256)
    p.add_argument("--samples_per_shard", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

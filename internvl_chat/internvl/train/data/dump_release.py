"""
Per-rank CLI to dump the DCVLM-Baseline pre-mixed release.

Launched one process per rank (via srun); RANK / WORLD_SIZE come from the
environment. Each rank spins a WebLoader with `--num_workers` CPU workers that
run the faithful production mix (see dump.py), consumes the merged stream, and
writes it to its own staging shards with a per-rank token budget
(`--global_token_budget / WORLD_SIZE`). No GPU / model / torchrun needed.

Idempotent: a rank that already has a `markers/r<rank>.done` file is skipped.
On restart of an unfinished rank, delete its `staging/r<rank>-*.tar` first — the
stream is deterministic in (seed, rank, worker), so a re-run reproduces it.

Run:
  RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS \
  python -m internvl.train.data.dump_release \
    --meta_path $META_DIR/large-pool/10c-5m-15t-70i.json \
    --output_folder $OUT --global_token_budget 200e9 \
    --num_workers 8 --samples_per_shard 10000 --seed 40 --num_images_expected 24

Finalize (single process) with finalize_release.py once all ranks are done.
"""

import os
import json
import time
import argparse
from types import SimpleNamespace

import webdataset as wds

from .dump import build_dump_stream, sample_to_output


def build_data_args(args) -> SimpleNamespace:
    """Minimal data_args carrying only the fields build_dump_stream reads.
    Values match the production run (scripts/slurm/train/run.sh); the image
    hyperparams are inert under image_formatter(skip_transform=True) except
    max_dynamic_patch, which is written as provenance (`mdp.txt`)."""
    return SimpleNamespace(
        meta_path=args.meta_path,
        num_image_token=256,          # unused under skip_transform
        force_image_size=448,
        pad2square=False,
        use_thumbnail=True,
        dynamic_image_size=True,
        min_dynamic_patch=1,
        max_dynamic_patch=args.max_dynamic_patch,
        normalize_type="imagenet",
        conv_style="internvl2_5",
    )


def main(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_rank_budget = float(args.global_token_budget) / world_size

    staging_dir = os.path.join(args.output_folder, "staging")
    markers_dir = os.path.join(args.output_folder, "markers")
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)

    prefix = f"r{rank:05d}"
    done_marker = os.path.join(markers_dir, f"{prefix}.done")
    if os.path.exists(done_marker):
        print(f"[dump] rank {rank}/{world_size}: marker exists, skipping.", flush=True)
        return

    collate_dtypes = tuple(d for d in args.collate_dtypes.split(",") if d) if args.collate_dtypes else ()
    data_args = build_data_args(args)

    stream = build_dump_stream(
        data_args,
        meta_path=args.meta_path,
        seed=args.seed,
        num_images_expected=args.num_images_expected,
        collate_dtypes=collate_dtypes,
        mmdoc_format=args.mmdoc_format,
        allow_distributed_replacement=args.allow_distributed_replacement,
    )

    loader_kwargs = dict(
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
    )
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = wds.WebLoader(stream, **loader_kwargs)

    sink_pattern = os.path.join(staging_dir, f"{prefix}-%06d.tar")
    cum_tokens = 0.0
    n_samples = 0
    n_skipped = 0
    per_dtype = {}
    t0 = time.time()
    stop = False

    print(
        f"[dump] rank {rank}/{world_size}: budget={per_rank_budget:.3e} tok, "
        f"meta={args.meta_path}, workers={args.num_workers}",
        flush=True,
    )

    with wds.ShardWriter(sink_pattern, maxcount=args.samples_per_shard, maxsize=10e10) as writer:
        for sample in loader:
            try:
                out, ntok = sample_to_output(sample, key=f"{prefix}_{n_samples:09d}")
            except Exception as e:
                n_skipped += 1
                if n_skipped <= 20:
                    print(f"[dump] rank {rank}: skipped sample ({type(e).__name__}: {e})", flush=True)
                continue

            writer.write(out)
            cum_tokens += ntok
            n_samples += 1
            dt = out.get("data_type.txt", "")
            per_dtype[dt] = per_dtype.get(dt, 0) + 1

            if n_samples % (10 * args.samples_per_shard) == 0:
                frac = cum_tokens / per_rank_budget if per_rank_budget else 0.0
                print(
                    f"[dump] rank {rank}: {n_samples:,} samples, "
                    f"{cum_tokens:.3e}/{per_rank_budget:.3e} tok ({frac:.1%})",
                    flush=True,
                )

            if cum_tokens >= per_rank_budget:
                break

    payload = {
        "rank": rank,
        "world_size": world_size,
        "n_samples": n_samples,
        "n_skipped": n_skipped,
        "cum_tokens": cum_tokens,
        "per_dtype": per_dtype,
        "seed": args.seed,
        "meta_path": args.meta_path,
        "seconds": time.time() - t0,
    }
    tmp_marker = done_marker + ".tmp"
    with open(tmp_marker, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_marker, done_marker)

    print(
        f"[dump] rank {rank}: DONE — {n_samples:,} samples "
        f"({cum_tokens:.3e} tok, {n_skipped} skipped) in {time.time()-t0:.0f}s",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Dump the DCVLM-Baseline pre-mixed release (per rank).")
    p.add_argument("--meta_path", type=str, required=True, help="Absolute path to the mix meta-JSON.")
    p.add_argument("--output_folder", type=str, required=True, help="Release root (holds staging/ and markers/).")
    p.add_argument("--global_token_budget", type=float, required=True, help="Total token budget across all ranks (e.g. 200e9).")
    p.add_argument("--seed", type=int, default=40, help="Mixing seed (fixed for reproducibility).")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers per rank.")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--samples_per_shard", type=int, default=10_000)
    p.add_argument("--num_images_expected", type=int, default=24, help="max_images cap (matches training PACKED_IMAGES).")
    p.add_argument("--max_dynamic_patch", type=int, default=12, help="Provenance only (written to mdp.txt).")
    p.add_argument("--collate_dtypes", type=str, default="instruction", help="Comma-separated dtypes to collate (match training).")
    p.add_argument("--mmdoc_format", type=str, default="mt", choices=["mt", "ic"])
    p.add_argument("--allow_distributed_replacement", type=lambda s: s.lower() in ("1", "true", "yes"), default=True,
                   help="Replacement sampling (True = match production; duplicates intended).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

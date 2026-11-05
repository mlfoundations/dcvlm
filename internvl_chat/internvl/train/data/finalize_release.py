"""
Finalize a DCVLM-Baseline pre-mixed release (single process, run after all ranks).

  1. Verify every expected `markers/r<rank>.done` is present (abort + list missing).
  2. Deterministically order `staging/r*-*.tar` and `os.rename` them (same-FS,
     metadata-only) into a flat `release/000000.tar … NNNNNN.tar`.
  3. Emit `release/dataset_info.json` (totals, seed, meta hash, per-data_type counts)
     and `release/meta_flat.json` — a single-entry, MAmmoTH-VL-shaped meta the trainer
     can consume directly.

Run:
  python -m internvl.train.data.finalize_release --output_folder $OUT
"""

import os
import json
import glob
import hashlib
import argparse


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    out = args.output_folder
    staging_dir = os.path.join(out, "staging")
    markers_dir = os.path.join(out, "markers")
    release_dir = os.path.join(out, "release")

    # 1. verify markers ----------------------------------------------------
    markers = sorted(glob.glob(os.path.join(markers_dir, "r*.done")))
    if not markers:
        raise SystemExit(f"No .done markers found in {markers_dir}; nothing to finalize.")

    payloads = []
    for m in markers:
        with open(m) as f:
            payloads.append(json.load(f))

    world_size = payloads[0]["world_size"]
    seen_ranks = {p["rank"] for p in payloads}
    missing = [r for r in range(world_size) if r not in seen_ranks]
    if missing:
        raise SystemExit(
            f"Incomplete: {len(missing)}/{world_size} ranks missing markers: "
            f"{missing[:50]}{' ...' if len(missing) > 50 else ''}"
        )
    print(f"[finalize] all {world_size} rank markers present.")

    # 2. flat-rename staging shards ---------------------------------------
    staged = sorted(glob.glob(os.path.join(staging_dir, "r*-*.tar")))
    if not staged:
        raise SystemExit(f"No staged shards found in {staging_dir}.")
    os.makedirs(release_dir, exist_ok=True)

    n_shards = len(staged)
    width = max(6, len(str(n_shards - 1)))
    for i, src in enumerate(staged):
        dst = os.path.join(release_dir, f"{i:0{width}d}.tar")
        os.rename(src, dst)
    print(f"[finalize] renamed {n_shards} shards -> {release_dir}/{'0'*width}.tar .. {n_shards-1:0{width}d}.tar")

    # 3. aggregate stats + write dataset_info.json / meta_flat.json --------
    total_samples = sum(p["n_samples"] for p in payloads)
    total_tokens = sum(p["cum_tokens"] for p in payloads)
    total_skipped = sum(p.get("n_skipped", 0) for p in payloads)
    per_dtype = {}
    for p in payloads:
        for dt, c in p.get("per_dtype", {}).items():
            per_dtype[dt] = per_dtype.get(dt, 0) + c
    per_dtype_frac = {dt: (c / total_samples if total_samples else 0.0) for dt, c in per_dtype.items()}

    meta_path = payloads[0].get("meta_path")
    meta_hash = _sha256_file(meta_path) if meta_path and os.path.exists(meta_path) else None
    seed = payloads[0].get("seed")

    first = f"{0:0{width}d}"
    last = f"{n_shards - 1:0{width}d}"
    release_root_brace = os.path.join(release_dir, "{" + f"{first}..{last}" + "}.tar")

    dataset_info = {
        "name": "dcvlm_baseline_flat",
        "source_meta": meta_path,
        "source_meta_sha256": meta_hash,
        "seed": seed,
        "world_size": world_size,
        "num_shards": n_shards,
        "num_samples": total_samples,
        "num_samples_skipped": total_skipped,
        "total_length_tokens": total_tokens,
        "per_data_type_counts": per_dtype,
        "per_data_type_fraction": per_dtype_frac,
        "release_root": release_root_brace,
    }
    with open(os.path.join(release_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    meta_flat = {
        "dcvlm_baseline_flat": {
            "root": release_root_brace,
            "data_type": "instruction",
            "data_augment": False,
            "repeat_time": 1,
            "length": 1.0,
            "dataloading_class": "webdataset",
            "filters": {},
            "subject_to": {},
        }
    }
    with open(os.path.join(release_dir, "meta_flat.json"), "w") as f:
        json.dump(meta_flat, f, indent=2)

    print(f"[finalize] {total_samples:,} samples, {total_tokens:.3e} tokens, {n_shards} shards.")
    print(f"[finalize] per-data_type fractions: "
          + ", ".join(f"{dt}={frac:.3f}" for dt, frac in sorted(per_dtype_frac.items())))
    print(f"[finalize] wrote dataset_info.json + meta_flat.json to {release_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Finalize the pre-mixed release (rename shards + write meta).")
    p.add_argument("--output_folder", type=str, required=True, help="Release root (holds staging/ and markers/).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

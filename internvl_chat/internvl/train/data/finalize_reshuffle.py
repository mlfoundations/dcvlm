"""
Finalize a globally-reshuffled release (single process, run after gather).

  1. Verify all scatter markers (WORLD_SIZE ranks) and all gather markers
     (num_buckets) are present; cross-check total_scattered == total_gathered ==
     the original release's num_samples (no samples lost or duplicated).
  2. os.rename `shuffled/b*-*.tar` -> flat `release_shuffled/000000.tar .. NNNNNN.tar`
     (same-FS, metadata-only). Output shard order is randomized by shard filename
     hash so even the concatenation of buckets is order-independent.
  3. Emit `release_shuffled/dataset_info.json` (carrying the original token / per-
     data_type totals, marked globally_shuffled) + `meta_flat.json`.

Run:
  python -m internvl.train.data.finalize_reshuffle \
    --root $OUT/reshuffle --orig_release $OUT/release
"""

import os
import json
import glob
import hashlib
import argparse

from .reshuffle import SCATTER_MARKERS, GATHER_MARKERS, GATHER_DIRNAME


def main(args):
    root = args.root
    gather_dir = os.path.join(root, GATHER_DIRNAME)
    scatter_markers = sorted(glob.glob(os.path.join(root, SCATTER_MARKERS, "r*.done")))
    gather_markers = sorted(glob.glob(os.path.join(root, GATHER_MARKERS, "b*.done")))
    if not scatter_markers:
        raise SystemExit(f"No scatter markers in {root}/{SCATTER_MARKERS}")
    if not gather_markers:
        raise SystemExit(f"No gather markers in {root}/{GATHER_MARKERS}")

    # 1. verify completeness + conservation --------------------------------
    scat = [json.load(open(m)) for m in scatter_markers]
    gath = [json.load(open(m)) for m in gather_markers]

    # world_size is read from the first scatter marker; markers from a restarted
    # run with a different task count are accepted as long as the union of all
    # ranked markers covers [0, world_size) and totals are conserved.
    world_size = scat[0]["world_size"]
    num_buckets = scat[0]["num_buckets"]
    seen_ranks = {p["rank"] for p in scat}
    missing_r = [r for r in range(world_size) if r not in seen_ranks]
    if missing_r:
        raise SystemExit(f"scatter incomplete: {len(missing_r)}/{world_size} ranks missing: {missing_r[:50]}")
    seen_buckets = {p["bucket"] for p in gath}
    missing_b = [b for b in range(num_buckets) if b not in seen_buckets]
    if missing_b:
        raise SystemExit(f"gather incomplete: {len(missing_b)}/{num_buckets} buckets missing: {missing_b[:50]}")

    total_scattered = sum(p["n_scattered"] for p in scat)
    total_gathered = sum(p["n_gathered"] for p in gath)
    if total_scattered != total_gathered:
        raise SystemExit(f"CONSERVATION FAIL: scattered {total_scattered:,} != gathered {total_gathered:,}")

    orig_info_path = os.path.join(args.orig_release, "dataset_info.json")
    orig_info = json.load(open(orig_info_path)) if os.path.exists(orig_info_path) else {}
    orig_n = orig_info.get("num_samples")
    if orig_n is not None and orig_n != total_gathered:
        raise SystemExit(f"CONSERVATION FAIL vs original: orig {orig_n:,} != gathered {total_gathered:,}")
    print(f"[finalize-reshuffle] verified: {world_size} ranks, {num_buckets} buckets, "
          f"{total_gathered:,} samples conserved.")

    # 2. flat-rename gathered shards (order randomized by name hash) --------
    staged = glob.glob(os.path.join(gather_dir, "b*-*.tar"))
    if not staged:
        raise SystemExit(f"No gathered shards in {gather_dir}")
    staged.sort(key=lambda p: hashlib.sha1(os.path.basename(p).encode()).hexdigest())

    release_dir = os.path.join(root, "release_shuffled")
    os.makedirs(release_dir, exist_ok=True)
    n_shards = len(staged)
    width = max(6, len(str(n_shards - 1)))
    for i, src in enumerate(staged):
        os.rename(src, os.path.join(release_dir, f"{i:0{width}d}.tar"))
    print(f"[finalize-reshuffle] renamed {n_shards} shards -> {release_dir}/"
          f"{'0'*width}.tar .. {n_shards-1:0{width}d}.tar")

    # 3. dataset_info.json + meta_flat.json --------------------------------
    first, last = f"{0:0{width}d}", f"{n_shards-1:0{width}d}"
    release_root_brace = os.path.join(release_dir, "{" + f"{first}..{last}" + "}.tar")

    info = dict(orig_info)
    info.update({
        "name": orig_info.get("name", "dcvlm_baseline_flat") + "_shuffled",
        "num_shards": n_shards,
        "num_samples": total_gathered,
        "globally_shuffled": True,
        "reshuffle_num_buckets": num_buckets,
        "reshuffle_seed": scat[0]["seed"],
        "reshuffle_shuffle_bufsize": gath[0]["shuffle_bufsize"],
        "release_root": release_root_brace,
        "source_release": args.orig_release,
    })
    with open(os.path.join(release_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

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

    print(f"[finalize-reshuffle] {total_gathered:,} samples, {n_shards} shards -> {release_dir}")
    print(f"[finalize-reshuffle] wrote dataset_info.json + meta_flat.json (globally_shuffled=True)")
    print(f"[finalize-reshuffle] scatter/ can now be deleted: rm -rf {os.path.join(root, 'scatter')}")


def parse_args():
    p = argparse.ArgumentParser(description="Finalize the globally-reshuffled release.")
    p.add_argument("--root", required=True, help="Reshuffle work root (holds scatter/, shuffled/, markers).")
    p.add_argument("--orig_release", required=True, help="Original finalized release dir (for totals cross-check).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

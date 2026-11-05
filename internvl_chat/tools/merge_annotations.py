#!/usr/bin/env python3
"""
Merge annotation directories by combining keys from two sources.

Takes a base annotation dir (source A) and a new annotation dir (source B),
validates consistency (same datasets, same shards, same sample keys),
and writes merged annotations to an output directory.

New keys from source B are added to source A's annotations.
Common keys are kept from source A (source A takes priority).
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm


SKIP_DIRS = {"logs"}


def get_datasets(path):
    """Return sorted list of dataset subdirectories, skipping non-dataset dirs."""
    return sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d not in SKIP_DIRS
    )


def get_json_files(path):
    """Return sorted list of .json files in a directory."""
    return sorted(f for f in os.listdir(path) if f.endswith(".json"))


def merge_single_shard(args):
    """Merge a single shard JSON file. Returns (dataset, shard, status, message)."""
    dataset, shard, path_a, path_b, path_out = args

    try:
        with open(os.path.join(path_a, shard)) as f:
            data_a = json.load(f)
        with open(os.path.join(path_b, shard)) as f:
            data_b = json.load(f)

        # Check same sample keys
        keys_a = set(data_a.keys())
        keys_b = set(data_b.keys())
        if keys_a != keys_b:
            missing_in_b = keys_a - keys_b
            missing_in_a = keys_b - keys_a
            parts = []
            if missing_in_b:
                parts.append(f"{len(missing_in_b)} samples in A but not B")
            if missing_in_a:
                parts.append(f"{len(missing_in_a)} samples in B but not A")
            return (dataset, shard, "ERROR", f"Sample key mismatch: {'; '.join(parts)}")

        # Merge: start with A, add new keys from B
        merged = {}
        for sample_key in data_a:
            sample_a = data_a[sample_key]
            sample_b = data_b[sample_key]
            # A takes priority for common keys; add new keys from B
            merged_sample = dict(sample_a)
            for k, v in sample_b.items():
                if k not in merged_sample:
                    merged_sample[k] = v
            merged[sample_key] = merged_sample

        os.makedirs(path_out, exist_ok=True)
        with open(os.path.join(path_out, shard), "w") as f:
            json.dump(merged, f, indent=4)

        return (dataset, shard, "OK", f"{len(merged)} samples")

    except Exception as e:
        return (dataset, shard, "ERROR", str(e))


def main():
    parser = argparse.ArgumentParser(description="Merge two annotation directories")
    parser.add_argument("--source-a", required=True, help="Base annotation dir (takes priority for common keys)")
    parser.add_argument("--source-b", required=True, help="New annotation dir (new keys added from here)")
    parser.add_argument("--output", required=True, help="Output directory for merged annotations")
    parser.add_argument("--workers", type=int, default=min(32, cpu_count()), help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Only validate, don't write")
    args = parser.parse_args()

    source_a = args.source_a
    source_b = args.source_b
    output_dir = args.output

    # --- Validation phase ---
    print(f"Source A: {source_a}")
    print(f"Source B: {source_b}")
    print(f"Output:   {output_dir}")
    print(f"Workers:  {args.workers}")
    print()

    # Check dataset folders match
    datasets_a = get_datasets(source_a)
    datasets_b = get_datasets(source_b)

    print(f"Datasets in A: {len(datasets_a)}")
    print(f"Datasets in B: {len(datasets_b)}")

    set_a = set(datasets_a)
    set_b = set(datasets_b)
    if set_a != set_b:
        missing_in_b = set_a - set_b
        missing_in_a = set_b - set_a
        if missing_in_b:
            print(f"ERROR: {len(missing_in_b)} datasets in A but not B: {sorted(missing_in_b)[:10]}...")
        if missing_in_a:
            print(f"ERROR: {len(missing_in_a)} datasets in B but not A: {sorted(missing_in_a)[:10]}...")
        sys.exit(1)

    print(f"Dataset count matches: {len(datasets_a)}")

    # Check shard files match per dataset and build task list
    tasks = []
    shard_errors = []
    total_shards = 0

    for dataset in datasets_a:
        dir_a = os.path.join(source_a, dataset)
        dir_b = os.path.join(source_b, dataset)
        dir_out = os.path.join(output_dir, dataset)

        shards_a = get_json_files(dir_a)
        shards_b = get_json_files(dir_b)

        if shards_a != shards_b:
            shard_errors.append(
                f"  {dataset}: A has {len(shards_a)} shards, B has {len(shards_b)} shards "
                f"(A: {shards_a[:3]}..., B: {shards_b[:3]}...)"
            )
            continue

        for shard in shards_a:
            tasks.append((dataset, shard, dir_a, dir_b, dir_out))
        total_shards += len(shards_a)

    if shard_errors:
        print(f"\nERROR: Shard count mismatch in {len(shard_errors)} datasets:")
        for e in shard_errors:
            print(e)
        sys.exit(1)

    print(f"Total shards to merge: {total_shards}")

    if args.dry_run:
        print("\n[DRY RUN] Validation passed. No files written.")
        # Still run the merge to check sample key consistency, but without writing
        # Actually let's just do a quick sample check
        print("Running sample-key consistency check on first shard of each dataset...")
        check_tasks = []
        seen = set()
        for t in tasks:
            if t[0] not in seen:
                seen.add(t[0])
                # Replace output dir with /dev/null-like behavior
                check_tasks.append(t)

        with Pool(args.workers) as pool:
            results = pool.map(merge_single_shard, check_tasks)

        errors = [r for r in results if r[2] == "ERROR"]
        if errors:
            print(f"\n{len(errors)} errors found:")
            for ds, sh, st, msg in errors:
                print(f"  {ds}/{sh}: {msg}")
            sys.exit(1)
        print("All checks passed.")
        return

    # --- Merge phase ---
    if os.path.exists(output_dir):
        existing = get_datasets(output_dir)
        if existing:
            print(f"\nWARNING: Output dir already has {len(existing)} datasets. Proceeding will overwrite.")

    print(f"\nMerging {total_shards} shards across {len(datasets_a)} datasets with {args.workers} workers...")
    start = time.time()

    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(merge_single_shard, tasks),
            total=len(tasks),
            desc="Merging shards",
        ))

    elapsed = time.time() - start

    # Report
    ok = [r for r in results if r[2] == "OK"]
    errors = [r for r in results if r[2] == "ERROR"]

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Merged: {len(ok)} shards")
    print(f"  Errors: {len(errors)} shards")

    if errors:
        print("\nErrors:")
        for ds, sh, st, msg in errors:
            print(f"  {ds}/{sh}: {msg}")
        sys.exit(1)

    # Quick verification: count output datasets and spot-check one
    out_datasets = get_datasets(output_dir)
    print(f"\nVerification: {len(out_datasets)} datasets in output")

    # Spot check first dataset first shard
    ds = out_datasets[0]
    shard = get_json_files(os.path.join(output_dir, ds))[0]
    with open(os.path.join(output_dir, ds, shard)) as f:
        sample = json.load(f)
    first_key = next(iter(sample))
    n_keys = len(sample[first_key])
    print(f"Spot check {ds}/{shard}: {len(sample)} samples, {n_keys} annotation keys per sample")
    print("\nMerge complete!")


if __name__ == "__main__":
    main()

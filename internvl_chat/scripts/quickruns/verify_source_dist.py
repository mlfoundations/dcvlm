"""
Scaled per-source faithfulness verifier for a DCVLM-Baseline pre-mixed release.

check_release.py only validates the 4 coarse data_type buckets. This tool validates the
full 166-source mixing: it estimates each source's observed sampling fraction from the
per-sample `ds_name.txt` provenance and compares it to the source's expected mixing weight
(`length` in the source-pool meta).

WHY SAMPLE MANY SHARDS: a single tar shard is ~10k *consecutive* samples from one rank at
one stream position, so the samples within it are correlated (finite shuffle buffer). One
shard ~= one independent "window", not 10k independent draws. So per-source accuracy is set
by the NUMBER of shards sampled, not the sample count. Sampling many shards spread across
all 512 ranks / stream positions averages the window-to-window variance away.

Reads ONLY the small `ds_name.txt` / `data_type.txt` members (seeks past image blobs) and
parallelises across shards with multiprocessing, so it scales to hundreds of shards.

Usage (see verify_source_dist.sh for the srun wrapper):
  python verify_source_dist.py --release_dir <OUT>/release --meta <large-pool meta> \
      --num_shards 800 --num_workers 32 --seed 0 --out_png dist.png --out_csv dist.csv
"""
import os
import csv
import json
import glob
import random
import tarfile
import argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DTYPE_COLORS = {
    "instruction": "#4C78A8",
    "text": "#F58518",
    "captioning": "#54A24B",
    "mmdoc_mt": "#E45756",
    "mmdoc_ic": "#B279A2",
}
_HEAD_K = 0  # set by main; 0 = read the whole shard (unbiased), >0 = first K samples (fast)


def _scan_one(tar_path):
    """Return (Counter ds_name, Counter data_type, n_samples) for one shard,
    reading only the *.ds_name.txt / *.data_type.txt members."""
    ds = Counter()
    dt = Counter()
    n = 0
    try:
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name
                if name.endswith(".ds_name.txt"):
                    ds[tf.extractfile(m).read().decode("utf-8", "replace").strip()] += 1
                    n += 1
                    if _HEAD_K and n >= _HEAD_K:
                        break
                elif name.endswith(".data_type.txt"):
                    dt[tf.extractfile(m).read().decode("utf-8", "replace").strip()] += 1
    except Exception as e:
        print(f"  !! failed to read {os.path.basename(tar_path)}: {type(e).__name__}: {e}")
    return ds, dt, n


def _init_worker(head_k):
    global _HEAD_K
    _HEAD_K = head_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release_dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--num_shards", type=int, default=800,
                    help="number of shards to sample (each ~= one independent window)")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head_k", type=int, default=0,
                    help="0 = read whole shard (unbiased, slower); >0 = only first K samples/shard (fast)")
    ap.add_argument("--spread", action="store_true",
                    help="pick shards evenly spaced across all shards instead of a random subset")
    ap.add_argument("--tol", type=float, default=0.02, help="abs tolerance for the bucket check")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    exp_weight = {k: float(v.get("length", 0.0)) for k, v in meta.items()}
    src_dtype = {k: v.get("data_type", "?") for k, v in meta.items()}
    n_sources = len(meta)

    all_shards = sorted(glob.glob(os.path.join(args.release_dir, "*.tar")))
    if not all_shards:
        raise SystemExit(f"No shards in {args.release_dir}")
    ns = min(args.num_shards, len(all_shards))
    if args.spread:
        idx = np.linspace(0, len(all_shards) - 1, ns).round().astype(int)
        shards = [all_shards[i] for i in sorted(set(idx))]
    else:
        rng = random.Random(args.seed)
        shards = rng.sample(all_shards, ns)

    print(f"release   : {args.release_dir}")
    print(f"shards    : {len(shards)} / {len(all_shards)}  "
          f"({'spread' if args.spread else f'random seed={args.seed}'}, "
          f"{'whole-shard' if not args.head_k else f'head_k={args.head_k}'})")
    print(f"workers   : {args.num_workers}")

    src_counts = Counter()
    dt_counts = Counter()
    total = 0
    with Pool(args.num_workers, initializer=_init_worker, initargs=(args.head_k,)) as pool:
        done = 0
        for ds, dt, n in pool.imap_unordered(_scan_one, shards, chunksize=1):
            src_counts.update(ds)
            dt_counts.update(dt)
            total += n
            done += 1
            if done % 25 == 0 or done == len(shards):
                print(f"  scanned {done}/{len(shards)} shards, {total:,} samples so far")

    if total == 0:
        raise SystemExit("No samples read.")

    # ---- faithfulness metrics ----
    obs = {k: src_counts.get(k, 0) / total for k in meta}
    # total variation distance between observed and expected source distributions
    tv = 0.5 * sum(abs(obs[k] - exp_weight[k]) for k in meta)
    seen = [k for k in meta if src_counts.get(k, 0) > 0]
    unknown = [d for d in src_counts if d not in meta]
    exp_ge1 = sum(1 for k in meta if exp_weight[k] * total >= 1.0)

    print("\n" + "=" * 84)
    print(f"PER-SOURCE FAITHFULNESS  ({total:,} samples over {len(shards)} shards)")
    print("=" * 84)
    print(f"  sources present               : {len(seen)} / {n_sources}")
    print(f"  sources w/ >=1 expected draw  : {exp_ge1}  (rest too rare at {total:,} samples)")
    print(f"  total-variation distance      : {tv:.4f}   (0 = perfect; shrinks with more shards)")
    if unknown:
        print(f"  !! ds_name values NOT in meta : {unknown}")

    bucket = Counter()
    for d, c in dt_counts.items():
        bucket["mmdoc" if d.startswith("mmdoc") else d] += c
    tot_dt = sum(dt_counts.values()) or 1
    print("\n  data_type buckets (observed vs target):")
    for b, tgt in (("captioning", .10), ("mmdoc", .05), ("text", .15), ("instruction", .70)):
        frac = bucket.get(b, 0) / tot_dt
        flag = "" if abs(frac - tgt) <= args.tol else "  <-- OFF"
        print(f"    {b:12s}: {frac:.4f}  (target {tgt:.2f}){flag}")

    # worst deviations among sources expected to appear
    devs = sorted(((k, obs[k], exp_weight[k],
                    (obs[k] / exp_weight[k] if exp_weight[k] else float("inf")))
                   for k in meta if exp_weight[k] * total >= 1.0),
                  key=lambda r: abs(r[3] - 1.0), reverse=True)
    print("\n  10 largest obs/exp deviations (sources expected to appear):")
    print(f"    {'source':32s} {'obs':>9s} {'exp':>9s} {'obs/exp':>8s} {'n':>8s}")
    for k, o, e, r in devs[:10]:
        print(f"    {k:32s} {o:9.5f} {e:9.5f} {r:8.2f} {src_counts.get(k,0):8d}")

    # ---- CSV ----
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "data_type", "n", "obs_frac", "exp_weight", "obs_over_exp"])
        for k in sorted(meta, key=lambda k: exp_weight[k], reverse=True):
            e = exp_weight[k]
            w.writerow([k, src_dtype[k], src_counts.get(k, 0), f"{obs[k]:.6g}",
                        f"{e:.6g}", f"{obs[k]/e:.4g}" if e else ""])
    print(f"\n  CSV  -> {args.out_csv}")

    # ---- plot: (A) source histogram, (B) obs-vs-exp scatter, (C) obs/exp ratio ----
    present = sorted(seen, key=lambda k: src_counts[k], reverse=True)
    counts = [src_counts[k] for k in present]
    colsA = [DTYPE_COLORS.get(src_dtype[k], "#888") for k in present]

    fig = plt.figure(figsize=(20, max(10, 0.22 * len(present))))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.1, 1.1], wspace=0.35)

    # (A) histogram of samples per source
    axA = fig.add_subplot(gs[0, 0])
    y = np.arange(len(present))
    axA.barh(y, counts, color=colsA, height=0.82)
    axA.set_yticks(y); axA.set_yticklabels(present, fontsize=6)
    axA.invert_yaxis()
    axA.set_xscale("log")
    axA.set_xlabel("samples (log)")
    axA.set_title(f"Source histogram — {len(present)}/{n_sources} sources present\n"
                  f"{total:,} samples, {len(shards)} shards", fontsize=11)
    axA.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DTYPE_COLORS.values()]
    axA.legend(handles, list(DTYPE_COLORS.keys()), fontsize=8, loc="lower right")

    # (B) observed vs expected (log-log) with y=x diagonal
    axB = fig.add_subplot(gs[0, 1])
    ek = np.array([exp_weight[k] for k in meta if exp_weight[k] > 0])
    ok = np.array([max(obs[k], 1e-8) for k in meta if exp_weight[k] > 0])
    cb = [DTYPE_COLORS.get(src_dtype[k], "#888") for k in meta if exp_weight[k] > 0]
    axB.scatter(ek, ok, c=cb, s=22, alpha=0.8)
    lo, hi = ek.min() / 2, ek.max() * 2
    axB.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="obs = exp")
    axB.axhline(1.0 / total, color="grey", ls=":", lw=1, label=f"1-sample floor")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("expected weight"); axB.set_ylabel("observed fraction")
    axB.set_title(f"Faithfulness (TV dist = {tv:.4f})\npoints on line = faithful", fontsize=11)
    axB.legend(fontsize=8); axB.grid(alpha=0.3)

    # (C) obs/exp ratio for sources expected to appear
    axC = fig.add_subplot(gs[0, 2])
    ratio_src = [k for k in meta if exp_weight[k] * total >= 1.0]
    ratio_src.sort(key=lambda k: exp_weight[k], reverse=True)
    ratios = [(obs[k] / exp_weight[k]) if exp_weight[k] else 0 for k in ratio_src]
    colsC = [DTYPE_COLORS.get(src_dtype[k], "#888") for k in ratio_src]
    yc = np.arange(len(ratio_src))
    axC.barh(yc, ratios, color=colsC, height=0.82)
    axC.axvline(1.0, color="k", lw=1)
    axC.set_yticks(yc); axC.set_yticklabels(ratio_src, fontsize=6)
    axC.invert_yaxis()
    axC.set_xlabel("observed / expected")
    axC.set_title("Per-source ratio\n(1.0 = faithful)", fontsize=11)
    axC.grid(axis="x", alpha=0.3)

    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"  PLOT -> {args.out_png}")


if __name__ == "__main__":
    main()

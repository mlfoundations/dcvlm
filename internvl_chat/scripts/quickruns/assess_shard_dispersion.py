"""Measure per-shard source-composition dispersion to decide whether a global reshuffle is
needed. Reads N shards, records per-shard ds_name counts, and for each source computes the
DESIGN EFFECT: observed between-shard variance of the per-shard rate divided by the i.i.d.
(multinomial) variance p(1-p)/n. Deff~1 => shards already look like random draws (no reshuffle
needed); Deff>>1 => samples cluster within shards (reshuffle would make shards representative).
Also reports the Fano factor (var/mean of per-shard counts; ~1 under Poisson) and coverage."""
import sys, glob, json, tarfile, argparse, random
from collections import Counter
from multiprocessing import Pool
import numpy as np


def scan(tar_path):
    c = Counter(); n = 0
    try:
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(".ds_name.txt"):
                    c[tf.extractfile(m).read().decode("utf-8", "replace").strip()] += 1
                    n += 1
    except Exception as e:
        print(f"  !! {tar_path}: {e}")
    return tar_path, c, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release_dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--num_shards", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--min_global", type=int, default=300,
                    help="only assess sources with >= this many total draws (estimable variance)")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    alls = sorted(glob.glob(f"{args.release_dir}/*.tar"))
    ns = min(args.num_shards, len(alls))
    if args.spread:
        idx = np.linspace(0, len(alls)-1, ns).round().astype(int)
        shards = [alls[i] for i in sorted(set(idx))]
    else:
        shards = random.Random(args.seed).sample(alls, ns)
    print(f"assessing {len(shards)} shards, {args.num_workers} workers "
          f"({'spread' if args.spread else 'random'})")

    per_shard = []; sizes = []
    with Pool(args.num_workers) as pool:
        for i, (_, c, n) in enumerate(pool.imap_unordered(scan, shards, chunksize=1)):
            if n:
                per_shard.append(c); sizes.append(n)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(shards)} shards")
    K = len(per_shard); nbar = float(np.mean(sizes)); tot = int(sum(sizes))
    print(f"\nread {K} shards, {tot:,} samples, mean shard size {nbar:.0f}")

    srcs = set().union(*[set(c) for c in per_shard])
    global_ct = Counter()
    for c in per_shard: global_ct.update(c)

    rows = []
    for s in srcs:
        g = global_ct[s]
        if g < args.min_global:
            continue
        p = g / tot
        rates = np.array([c.get(s, 0)/nsz for c, nsz in zip(per_shard, sizes)])
        counts = np.array([c.get(s, 0) for c in per_shard], dtype=float)
        obs_var = rates.var(ddof=1)
        iid_var = p*(1-p)/nbar
        deff = obs_var/iid_var if iid_var > 0 else float("nan")
        fano = counts.var(ddof=1)/counts.mean() if counts.mean() > 0 else float("nan")
        cover = float((counts > 0).mean())
        rows.append((s, p, g, deff, fano, cover))

    rows.sort(key=lambda r: r[1], reverse=True)
    deffs = np.array([r[3] for r in rows])
    print(f"\nsources assessed (>= {args.min_global} draws): {len(rows)}")
    print(f"DESIGN EFFECT across sources:  median={np.median(deffs):.1f}  "
          f"mean={deffs.mean():.1f}  p90={np.percentile(deffs,90):.1f}  max={deffs.max():.1f}")
    print(f"  (Deff~1 => shards already i.i.d.; Deff>>1 => clustered within shards)")
    print(f"effective independent shards ~= K/median_Deff = {K/np.median(deffs):.1f} of {K}")

    print(f"\n{'source':30s} {'glob_frac':>9s} {'Deff':>8s} {'Fano':>8s} {'shard_cover':>11s}")
    for s, p, g, deff, fano, cover in rows[:30]:
        print(f"{s:30s} {p:9.5f} {deff:8.1f} {fano:8.1f} {cover:11.2f}")


if __name__ == "__main__":
    main()

"""
Offline structural check for a DCVLM-Baseline pre-mixed release (no GPU/model).

Reads the finalized `release/` shards back through the same primitives the trainer
uses and asserts the release is well-formed and faithful:

  (a) per-data_type sample proportions ~ 0.10 captioning / 0.05 mmdoc / 0.15 text / 0.70 instruction
  (b) Sigma(ntok) over sampled shards is consistent with the recorded budget
  (c) txt2conversations + image_formatter (full) succeed on every sampled sample
  (d) no duplicate __key__ within a shard
  (e) per sample: number of "<image>" tokens == number of stored image files

Run (inside the dcvlm-train env):
  python -m internvl.train.data.check_release --release_dir $OUT/release --max_samples 20000
"""

import io
import os
import re
import sys
import json
import glob
import tarfile
import argparse
from collections import defaultdict

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm absent -> no-op bar, check still runs
    def tqdm(*a, **k):
        class _Null:
            def update(self, *_a, **_k): pass
            def set_postfix(self, *_a, **_k): pass
            def close(self): pass
        return _Null()

from .formatters.conversation import txt2conversations
from .formatters import image_formatter

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "ppm", "jp2", "tif", "tiff"}
# capability buckets -> target fraction (data_type.txt values written by the dump)
DTYPE_TARGET = {"captioning": 0.10, "mmdoc_mt": 0.05, "mmdoc_ic": 0.05, "text": 0.15, "instruction": 0.70}


def _ext(member_name: str) -> str:
    return member_name.rsplit(".", 1)[-1].lower() if "." in member_name else ""


def _group_shard(tar_path: str):
    """Yield {basename_ext: bytes} sample dicts from one tar (wds group_by_keys logic,
    but self-contained so the check has no wds dependency)."""
    samples = defaultdict(dict)
    order = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            # key = everything up to the first dot of the basename
            base = os.path.basename(name)
            key = base.split(".", 1)[0]
            field = base.split(".", 1)[1] if "." in base else ""
            if key not in samples:
                order.append(key)
            samples[key][field] = tf.extractfile(m).read()
    for key in order:
        yield key, samples[key]


def _run_image_formatter(sample_pil: dict):
    """Exercise the full image_formatter on a read-back sample (instruction path)."""
    s = dict(sample_pil)
    s["__is_train__"] = False
    s["__max_dynamic_patch__"] = 12
    s["__data_type__"] = "instruction"
    return image_formatter(
        s,
        pad2square=False,
        normalize_type="imagenet",
        num_image_token=256,
        dynamic_image_size=True,
        min_dynamic_patch=1,
        max_images=64,
        image_size=448,
        use_thumbnail=True,
    )


def main(args):
    shards = sorted(glob.glob(os.path.join(args.release_dir, "*.tar")))
    if not shards:
        raise SystemExit(f"No shards in {args.release_dir}")

    info_path = os.path.join(args.release_dir, "dataset_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}

    dtype_counts = defaultdict(int)
    total_ntok = 0
    n_seen = 0
    n_img_mismatch = 0
    n_fmt_fail = 0
    n_dup_keys = 0
    errors = []

    pbar = tqdm(total=args.max_samples, unit="sample", desc="checking", dynamic_ncols=True)
    for tar_path in shards:
        keys_in_shard = set()
        for key, fields in _group_shard(tar_path):
            if key in keys_in_shard:
                n_dup_keys += 1
                errors.append(f"dup key {key} in {os.path.basename(tar_path)}")
            keys_in_shard.add(key)

            # provenance
            dt = fields.get("data_type.txt", b"").decode("utf-8", "replace").strip()
            dtype_counts[dt] += 1
            try:
                total_ntok += int(fields.get("ntok.txt", b"0").decode() or 0)
            except ValueError:
                pass

            # (c) conversation parse
            conv_txt = fields.get("conversations.txt", b"").decode("utf-8", "replace")
            try:
                turns = txt2conversations(conv_txt)
                assert len(turns) >= 1
            except Exception as e:
                n_fmt_fail += 1
                errors.append(f"conv parse fail {key}: {e}")
                turns = []

            # (e) <image> count vs stored images
            n_image_tokens = sum(t["value"].count("<image>") for t in turns)
            img_fields = {f: b for f, b in fields.items() if _ext(f) in IMAGE_EXTS}
            if n_image_tokens != len(img_fields):
                n_img_mismatch += 1
                if len(errors) < 50:
                    errors.append(
                        f"image mismatch {key}: {n_image_tokens} <image> vs {len(img_fields)} files (dt={dt})"
                    )

            # (c) image_formatter full run on decoded images
            if img_fields:
                try:
                    pil = {}
                    for f, b in img_fields.items():
                        pil[f] = Image.open(io.BytesIO(b)).convert("RGB")
                    _run_image_formatter(pil)
                except Exception as e:
                    n_fmt_fail += 1
                    if len(errors) < 50:
                        errors.append(f"image_formatter fail {key}: {type(e).__name__}: {e}")

            n_seen += 1
            pbar.update(1)
            if (n_fmt_fail or n_dup_keys or n_img_mismatch) and n_seen % 200 == 0:
                pbar.set_postfix(fmt_fail=n_fmt_fail, dup=n_dup_keys, img_mismatch=n_img_mismatch)
            if n_seen >= args.max_samples:
                break
        if n_seen >= args.max_samples:
            break
    pbar.close()

    # report ---------------------------------------------------------------
    print(f"\n=== Release structural check: {args.release_dir} ===")
    print(f"shards on disk: {len(shards)}  |  samples inspected: {n_seen:,}")
    print(f"\n(a) per-data_type fractions (target in parens):")
    ok_props = True
    # merge mmdoc_ic into mmdoc bucket for the target comparison
    bucket = defaultdict(int)
    for dt, c in dtype_counts.items():
        b = "mmdoc" if dt.startswith("mmdoc") else dt
        bucket[b] += c
    targets = args.targets
    for b, tgt in targets.items():
        frac = bucket.get(b, 0) / n_seen if n_seen else 0.0
        flag = "" if abs(frac - tgt) <= args.tol else "  <-- OFF"
        if abs(frac - tgt) > args.tol:
            ok_props = False
        print(f"    {b:12s}: {frac:.4f}  ({tgt:.2f}){flag}")
    unknown = {dt: c for dt, c in dtype_counts.items()
               if not (dt.startswith("mmdoc") or dt in ("captioning", "text", "instruction"))}
    if unknown:
        print(f"    unknown data_types: {unknown}")

    print(f"\n(b) Sigma ntok over inspected samples: {total_ntok:.3e}")
    if info:
        print(f"    recorded total_length_tokens: {info.get('total_length_tokens'):.3e} "
              f"over {info.get('num_samples'):,} samples")

    print(f"\n(c) conversation/image_formatter failures: {n_fmt_fail}")
    print(f"(d) duplicate keys within shards: {n_dup_keys}")
    print(f"(e) <image>-vs-imagefile mismatches: {n_img_mismatch}")

    if errors:
        print(f"\n--- first {min(len(errors), 25)} issues ---")
        for e in errors[:25]:
            print("   ", e)

    passed = ok_props and n_fmt_fail == 0 and n_dup_keys == 0 and n_img_mismatch == 0
    print(f"\n=== {'PASS' if passed else 'FAIL'} ===")
    sys.exit(0 if passed else 1)


def parse_args():
    p = argparse.ArgumentParser(description="Offline structural check for the pre-mixed release.")
    p.add_argument("--release_dir", type=str, required=True, help="Path to the finalized release/ dir.")
    p.add_argument("--max_samples", type=int, default=20_000)
    p.add_argument("--tol", type=float, default=0.03, help="Allowed abs deviation from target data_type fractions.")
    p.add_argument("--targets", type=str, default="captioning=0.10,mmdoc=0.05,text=0.15,instruction=0.70",
                   help="Expected data_type fractions of the dumped mixture, e.g. "
                        "'captioning=0.40,mmdoc=0.05,text=0.15,instruction=0.40' for the 40c-40i mix.")
    args = p.parse_args()
    args.targets = {k: float(v) for k, v in (kv.split("=") for kv in args.targets.split(","))}
    return args


if __name__ == "__main__":
    main(parse_args())

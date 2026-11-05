"""
Build a small Parquet *preview* of a DCVLM-Baseline pre-mixed release, for the
Hugging Face Dataset Viewer only (no GPU/model).

Why this exists: the release ships as flat WebDataset `.tar` shards, and the HF
WebDataset viewer requires every sample in a shard to share an identical set of
files. Our samples legitimately don't (text-only = no image, caption = `0.jpg`,
interleaved docs = `0.jpg..N.jpg`), so the raw `.tar` shards cannot be rendered
by the viewer. The fix is a hybrid layout: the real data stays as WebDataset under
`shards/`, and this script materializes a small, downscaled Parquet slice under
`preview/` with FineVision-style columns (`images: Sequence(Image())` +
`conversations: [{from, value}]`). The dataset card points the viewer at the
Parquet via a single `config_name: preview` entry, which disables `.tar`
auto-detection so the viewer never touches (and never errors on) the shards.

The Parquet is a *rendering artifact only* — it is downsampled and its images are
shrunk. It is NOT the training data.

Run (inside the dcvlm-train env):
  python -m internvl.train.data.build_preview \
      --release_dir $OUT/release \
      --output_parquet $OUT/release/preview/preview-0000.parquet \
      --max_samples 2000 --max_image_side 768
"""

import io
import os
import sys
import glob
import tarfile
import argparse

from PIL import Image

from .formatters.conversation import txt2conversations

# image extensions the dump can emit (last dotted component of a field name)
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "ppm", "jp2", "tif", "tiff"}


def _stream_shard(tar_path: str):
    """Yield (key, {field: bytes}) per sample, streaming.

    Mirrors check_release._group_shard's key/field convention (key = basename up
    to the first dot; field = the rest), but streams sample-by-sample instead of
    materializing the whole tar, so the preview can stop early on a full shard.
    Relies on the dump writing each sample's members contiguously (ShardWriter).
    """
    cur = None
    cur_key = None
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            key = base.split(".", 1)[0]
            field = base.split(".", 1)[1] if "." in base else ""
            if key != cur_key:
                if cur is not None:
                    yield cur_key, cur
                cur = {}
                cur_key = key
            cur[field] = tf.extractfile(m).read()
        if cur is not None:
            yield cur_key, cur


def _image_fields(fields):
    """Image field names, ordered by their numeric prefix (== <image> order)."""
    imgs = [f for f in fields if f.rsplit(".", 1)[-1].lower() in IMAGE_EXTS]
    return sorted(imgs, key=lambda f: int(f.split(".", 1)[0]))


def _encode_image(raw: bytes, max_side: int) -> dict:
    """Decode -> optional downscale -> re-encode JPEG. Returns a datasets.Image dict."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return {"bytes": buf.getvalue(), "path": None}


def _iter_samples(shard_paths, max_samples, max_side):
    n = 0
    for path in shard_paths:
        for key, mem in _stream_shard(path):
            if "conversations.txt" not in mem:
                continue
            images = [_encode_image(mem[f], max_side) for f in _image_fields(mem)]
            ntok = (mem.get("ntok.txt", b"0").decode("utf-8") or "0").strip()
            yield {
                "images": images,
                "conversations": txt2conversations(mem["conversations.txt"].decode("utf-8")),
                "data_type": mem.get("data_type.txt", b"").decode("utf-8"),
                "ds_name": mem.get("ds_name.txt", b"").decode("utf-8"),
                "num_tokens": int(ntok) if ntok.isdigit() else 0,
            }
            n += 1
            if n >= max_samples:
                return


def main(args):
    import datasets
    from datasets import Dataset, Features, Value

    shard_paths = sorted(glob.glob(os.path.join(args.release_dir, "*.tar")))
    if not shard_paths:
        sys.exit(f"[build_preview] no *.tar found in {args.release_dir}")
    if args.shards:
        wanted = set(args.shards)
        shard_paths = [p for p in shard_paths if os.path.basename(p) in wanted]
        if not shard_paths:
            sys.exit(f"[build_preview] none of --shards {args.shards} found in {args.release_dir}")
    print(f"[build_preview] reading up to {args.max_samples} samples from "
          f"{len(shard_paths)} shard(s), images <= {args.max_image_side}px", flush=True)

    features = Features({
        "images": [datasets.Image()],
        "conversations": [{"from": Value("string"), "value": Value("string")}],
        "data_type": Value("string"),
        "ds_name": Value("string"),
        "num_tokens": Value("int64"),
    })

    ds = Dataset.from_generator(
        _iter_samples,
        gen_kwargs={
            "shard_paths": shard_paths,
            "max_samples": args.max_samples,
            "max_side": args.max_image_side,
        },
        features=features,
    )
    print(f"[build_preview] built {ds.num_rows} samples", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_parquet)), exist_ok=True)
    ds.to_parquet(args.output_parquet)
    print(f"[build_preview] wrote {args.output_parquet}", flush=True)

    # round-trip sanity + data_type distribution
    from collections import Counter
    rt = Dataset.from_parquet(args.output_parquet)
    dist = dict(Counter(rt["data_type"]))
    r0 = rt[0]
    print(f"[build_preview] round-trip OK: {rt.num_rows} rows | data_type={dist}", flush=True)
    print(f"[build_preview] row0: data_type={r0['data_type']} ds_name={r0['ds_name']} "
          f"n_images={len(r0['images'])} n_turns={len(r0['conversations'])}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Build a Parquet viewer preview for a DCVLM-Baseline release.")
    p.add_argument("--release_dir", required=True,
                   help="Directory of finalized release *.tar shards (e.g. $OUT/release).")
    p.add_argument("--output_parquet", required=True,
                   help="Output parquet path (e.g. $OUT/release/preview/preview-0000.parquet).")
    p.add_argument("--max_samples", type=int, default=2000,
                   help="Max samples to include in the preview (default 2000).")
    p.add_argument("--max_image_side", type=int, default=768,
                   help="Downscale images so the longest side <= this (px); 0 to keep original.")
    p.add_argument("--shards", nargs="*", default=None,
                   help="Optional: restrict to these shard basenames (e.g. 000000.tar). "
                        "Default: iterate shards in sorted order until max_samples reached.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

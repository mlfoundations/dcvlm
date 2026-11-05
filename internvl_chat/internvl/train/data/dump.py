"""
Dataset-dump library for the DCVLM-Baseline pre-mixed release.

This builds the *exact* production training data stream (per-source filtering +
`DeterministicRandomMix` with the same params as `scripts/slurm/train/run.sh`),
taps it right after the per-type `conversation_formatter`, and exposes helpers
to serialize each drawn sample back into the MAmmoTH-VL-style on-disk schema:

    <key>.conversations.txt   # role:value turns, EOL_SEPARATOR("<EOCL>\\n") delimited
    <key>.0.<ext> ...         # original image bytes, in <image>-token order
    <key>.data_type.txt / ds_name.txt / orig_url.txt / mdp.txt / ntok.txt  # provenance

Why this is faithful by construction:
  * same byte pipelines + mix + seeding as training (replacement ON → intended
    duplicates from instruction oversampling are materialized);
  * the tap is *after* the correct per-type formatter, per drawn instance, so
    formatter randomness (caption prompt / mmdoc delimiters) is re-rolled per draw;
  * re-serializing the formatter's `conversations` list into `conversations.txt`
    means a flat single-source read dispatches every sample to `format_instruction`
    -> `txt2conversations` -> the identical `conversations` list (see conversation.py).

Images are written as their ORIGINAL bytes (captured before decode) so the release
is lossless; `image_formatter(skip_transform=True)` is only used to reproduce the
sort / mmdoc-trailing-image-drop / image-count that affect the conversation.
"""

import io
import logging
from functools import partial

import webdataset as wds
from PIL import Image

from internvl.train.constants import EOL_SEPARATOR
from .pipelinev2 import (
    build_all_pipelines,
    _SAMPLE_SHUFFLE_SIZE,
    _SAMPLE_SHUFFLE_INITIAL,
)
from .mix import DeterministicRandomMix
from .formatters import image_formatter, conversation_formatter
from .utils import log_and_continue_with_prefix

logger = logging.getLogger()

# file extensions decoded to PIL images by wds.decode("pilrgb"); we stash the
# raw bytes of these keys before decode replaces them with PIL objects.
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "ppm", "jp2", "tif", "tiff"}

# marker whose presence in a turn value would break the conversations.txt round-trip
_EOCL_MARKER = EOL_SEPARATOR.strip()  # "<EOCL>"


def _key_ext(key: str) -> str:
    # webdataset field keys are the filename minus the sample key, so a
    # single-image instruction/caption sample yields a *bare* extension field
    # (member `7.jpg` -> key `7`, field `jpg`) while mmdoc yields a numbered
    # field (member `..12.jpg` -> field `12.jpg`). Take the last dotted
    # component either way; a dot-less key returns itself (e.g. `jpg` -> `jpg`).
    return key.rsplit(".", 1)[-1].lower()


def stash_raw_media(sample: dict) -> dict:
    """Copy raw image bytes into a private key BEFORE `wds.decode` turns them into
    PIL objects, so we can write the original (lossless) bytes to the release."""
    raw = {}
    for k, v in sample.items():
        if k.startswith("__"):
            continue
        if isinstance(v, (bytes, bytearray)) and _key_ext(k) in IMAGE_EXTS:
            raw[k] = bytes(v)
    sample["__raw_media__"] = raw
    return sample


def build_dump_stream(
    data_args,
    meta_path: str,
    seed: int,
    num_images_expected: int,
    collate_dtypes=("instruction",),
    mmdoc_format: str = "mt",
    allow_distributed_replacement: bool = True,
):
    """Return a wds pipeline yielding fully mixed + per-type-formatted samples
    (with raw image bytes preserved), ready to be serialized to the release schema.

    Mirrors `build_data_mix` up to (and including) `conversation_formatter`, but:
      * stashes raw image bytes before decode,
      * runs `image_formatter(skip_transform=True)` (structure only, no pixel_values),
      * skips `huggingface_formatter` (no tokenization needed; token budget uses the
        precomputed per-sample `__length_tokens__`).
    """
    sources, probs, names, metas = build_all_pipelines(
        data_args,
        meta_path=meta_path,
        allow_distributed_replacement=allow_distributed_replacement,
        seed=seed,
        collate_dtypes=list(collate_dtypes),
        mmdoc_format=mmdoc_format,
        stash_length_tokens=True,
    )

    # normalize probs exactly as build_data_mix does
    probs = [p / sum(probs) for p in probs]

    mix = DeterministicRandomMix(sources, seed=seed, probs=probs, longest=False)

    im_formatter = partial(
        image_formatter,
        pad2square=data_args.pad2square,
        normalize_type=data_args.normalize_type,
        num_image_token=data_args.num_image_token,
        dynamic_image_size=data_args.dynamic_image_size,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_images=num_images_expected,
        image_size=data_args.force_image_size,
        use_thumbnail=data_args.use_thumbnail,
        skip_transform=True,
    )
    conv_formatter = conversation_formatter

    decode_handler = partial(log_and_continue_with_prefix, prefix="dump_decode_fn")
    im_handler = partial(log_and_continue_with_prefix, prefix="dump_im_formatter")
    conv_handler = partial(log_and_continue_with_prefix, prefix="dump_conv_formatter")

    return wds.DataPipeline(
        # weighted mix of raw-byte samples (post per-source filtering) -- the tap
        mix,
        # global sample shuffle (same buffer as training) for shard-level homogeneity
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
            seed=seed,
        ),
        # preserve original image bytes before decode overwrites them with PIL
        wds.map(stash_raw_media),
        # decode images -> PIL, text -> str (needed by the formatters below)
        wds.decode("pilrgb", handler=decode_handler),
        # structure-only image formatting (sort + mmdoc drop + count); no pixel_values
        wds.map(im_formatter, handler=im_handler),
        # per-type conversation formatting -> sample["conversations"] = [{from,value}]
        wds.map(conv_formatter, handler=conv_handler),
    )


def serialize_conversations(turns: list) -> str:
    """Inverse of `txt2conversations` (conversation.py): a [{from,value},...] list
    -> "role:value" joined by EOL_SEPARATOR. Raises if a value contains the EOCL
    marker (would corrupt the round-trip)."""
    parts = []
    for t in turns:
        value = t["value"]
        if _EOCL_MARKER in value:
            raise ValueError(
                f"conversation value contains '{_EOCL_MARKER}' marker; "
                "cannot serialize losslessly"
            )
        parts.append(f"{t['from']}:{value}")
    return EOL_SEPARATOR.join(parts)


def sample_to_output(sample: dict, key: str):
    """Convert a formatted dump-stream sample into a ShardWriter output dict
    following the release schema, plus its token length for budget accounting.

    Returns (out_dict, length_tokens).
    """
    raw_media = sample.get("__raw_media__", {})
    out = {"__key__": key}

    # collect surviving images in sample (= sorted / post-drop) order so the Nth
    # <image> token maps to the Nth written image
    img_idx = 0
    for k, v in sample.items():
        if k.startswith("__"):
            continue
        if isinstance(v, Image.Image):
            ext = _key_ext(k) or "jpg"
            if k in raw_media:
                out[f"{img_idx}.{ext}"] = raw_media[k]
            else:
                # fallback: original bytes missing (unexpected) -> re-encode losslessly
                buf = io.BytesIO()
                v.save(buf, format="PNG")
                out[f"{img_idx}.png"] = buf.getvalue()
                logger.error(
                    f"[dump] raw bytes missing for image key '{k}' "
                    f"(sample {sample.get('__ds_name__')}); re-encoded as PNG."
                )
            img_idx += 1

    out["conversations.txt"] = serialize_conversations(sample["conversations"])
    out["data_type.txt"] = str(sample.get("__data_type__", ""))
    out["ds_name.txt"] = str(sample.get("__ds_name__", ""))
    out["orig_url.txt"] = str(sample.get("__url__", ""))
    out["mdp.txt"] = str(sample.get("__max_dynamic_patch__", ""))
    length_tokens = int(sample.get("__length_tokens__", 0) or 0)
    out["ntok.txt"] = str(length_tokens)

    return out, length_tokens

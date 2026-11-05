# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# special character used to delimit
# conversation turns, allows for lossless decoding
# of wds instruction-tuning datasets
EOL_SEPARATOR = "<EOCL>\n"

# used to format image-caption pairs in the ChatML style
LLAVA_DESC_SHORT_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo’s key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]

# mapping from dataset name to a ChatML formatting function
# this is *NOT* used by default, but used as a last resort fallback
# if there is no formatter specified for the "data_type" key
DATASET_FORMAT_ROUTERS = {
    "relaion_2b": "format_icpairs",
    "grit_caption": "format_icpairs",
    "mint_html": "format_mmdoc_ic",
    "mint_pdf": "format_mmdoc_ic",
    "multimodal_textbook": "format_mmdoc_ic",
    "wanjuan": "format_mmdoc_ic",
    "as100m": "format_icpairs",  # this might be used for grounding as well, but needs impl.
    "omnicc": "format_mmdoc_ic",
    "datacomp_1b": "format_icpairs",
    "recap1b": "",  # not using this atm
    "internvl_sa1b_caption": "format_icpairs",  # ".txt" keys instead of "txt", watch out!
    "facecaption15m": "format_icpairs",
}

# fn operators for online filtering
# Allowed comparators for online filtering
import operator as _op
NUMERICAL_COMPARATORS = {
    ">":  _op.gt,
    ">=": _op.ge,
    "<":  _op.lt,
    "<=": _op.le,
    "==": _op.eq,
}
MEMBERSHIP_COMPARATORS = {
    "in": lambda x, y: x in y,
    "not in": lambda x, y: x not in y,
}

MMDOC_FORMATS = set([
    "mt",
    "ic"
])
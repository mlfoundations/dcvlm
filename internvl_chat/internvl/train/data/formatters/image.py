import torch
from PIL import Image
from collections import OrderedDict
from typing import Dict, Literal, Optional, Callable
from internvl.train.dataset import build_transform, dynamic_preprocess


def get_num_images_from_sample(sample: dict) -> int:
    c = 0
    for samp_key, value in sample.items():
        if isinstance(value, Image.Image):
            c += 1
    return c


def image_formatter(
    sample: dict,
    pad2square: bool,
    normalize_type: str,
    num_image_token: int,
    dynamic_image_size: bool,
    min_dynamic_patch: int,
    max_images: int,
    image_size: int,
    use_thumbnail: bool,
    skip_transform: bool = False
):
    '''
    This function handles the dynamic preprocessing of images as required by InternVL-2.5
    Input: dictionary from webdataset
    Output: dictionary of webdataset augmented with information about the dynamically preprocessed images

    When `skip_transform=True`, only the sample-structuring steps are performed
    (numeric-key sort, mmdoc trailing-image drop, `num_image` count) and the
    expensive pixel-value transform is skipped. Used by the dataset-dump tooling,
    which needs the same conversation-affecting structure as training but does not
    need `pixel_values`. The `num_image==0` blank-image insertion is also skipped
    so text-only samples carry no image (regenerated on read-back).
    '''

    tiles_to_return, num_tiles_per_image = [], []

    is_train = sample["__is_train__"]
    max_dynamic_patch = sample["__max_dynamic_patch__"]
    data_type = sample["__data_type__"]

    transform = build_transform(
        is_train=is_train, 
        input_size=image_size,
        pad2square=pad2square, 
        normalize_type=normalize_type
    )
    
    # We need to ensure that the `pixel_values` 
    # that are fed in for the mmdocs case are 
    # correctly sorted based on the position
    # of the image in the multimodal sequence,
    # hence we sort to be extra sure that order
    # is preserved.
    if data_type in ["mmdoc_mt", "mmdoc_ar", "mmdoc_ic", "instruction"]:
        sample = OrderedDict(sorted(
            sample.items(),
            key=lambda item: int(item[0].split('.')[0]) if item[0].split('.')[0].isdigit() else float('inf')
        ))

    if data_type in ["mmdoc_mt", "mmdoc_ar", "mmdoc_ic"]:
        # If the last item in the sequence is an image, we drop it since we cannot have loss over the last image.
        pop_from_last = []
        for samp_key, value in reversed(sample.items()):
            if not samp_key.split(".")[0].isdigit():
                continue
            if isinstance(value, Image.Image):
                pop_from_last.append(samp_key)
            else:
                break
        for k in pop_from_last:
            sample.pop(k)

    ### NOTE (15/05): we move this here to ensure that we get the correct number of images after the cleaning step for multimodal documents
    num_image = get_num_images_from_sample(sample)

    # if we hit a sample with more images than allowed,
    # we raise (the handler will log and continue)
    if num_image > max_images:
        raise ValueError(
            f"Sample has too many images ({num_image} > {max_images})."
        )

    # dump path: return the sorted / trailing-image-dropped sample with just the
    # image count, skipping the blank-image insert and the pixel transform below.
    if skip_transform:
        ret_dict = sample
        ret_dict['num_image'] = num_image
        return ret_dict

    # FIXME: if we don't have images, it means we hit a text-only webdataset
    # so we currently get around this by creating a blank image
    if num_image == 0:
        sample["jpg"] = Image.new('RGB', (224, 224), (255, 255, 255))

    # Iterate over all possible images in the sequence
    for samp_key, value in sample.items():
        # this works because we decode with pilrgb
        if isinstance(value, Image.Image):
            if dynamic_image_size:
                # handle text-only cases
                max_num = max(1, max_dynamic_patch // num_image) if num_image > 0 else 1
                curr_tiles = dynamic_preprocess(
                    value, 
                    min_num=min_dynamic_patch,
                    max_num=max_num,
                    image_size=image_size,
                    use_thumbnail=use_thumbnail
                )
                tiles_to_return += curr_tiles
                num_tiles_per_image.append(len(curr_tiles))
            else:
                tiles_to_return.append(value)
                num_tiles_per_image.append(1)

    # assert the code for image transforms works smoothly
    error_msg = "Tiles to return and total num tiles per image do not match, "
    error_msg += f"{sum(num_tiles_per_image)=}, {len(tiles_to_return)=}"
    assert sum(num_tiles_per_image) == len(tiles_to_return), error_msg

    ret_dict = sample

    # Get all the transformed tiles
    pixel_values = [transform(tile) for tile in tiles_to_return]
    pixel_values = torch.stack(pixel_values)
    total_num_tiles = pixel_values.size(0)

    # Preprocess the conversations and generate the return dictionary
    num_image_tokens_per_image = [num_image_token * num_tile for num_tile in num_tiles_per_image]

    # tensor of all stacked tiles across all images
    ret_dict['pixel_values'] = pixel_values  
    # list of num tokens per image, across images in the sequence
    ret_dict['num_image_tokens_per_image'] = num_image_tokens_per_image 
    # list of number of tiles per image in the sequence
    ret_dict['num_tiles_per_image'] = num_tiles_per_image 
    # the total number of tiles across all images
    ret_dict['total_num_tiles'] = total_num_tiles
    # total number of raw images (for mmdocs -> after discarding)
    ret_dict['num_image'] = num_image 

    return ret_dict
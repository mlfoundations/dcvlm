import io

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import torch
import os
import torch.nn.functional as F
import torchvision.transforms as T
import transformers
import numpy as np

from typing import Dict
from PIL import Image
from copy import deepcopy
from torchvision.transforms.functional import InterpolationMode

try:
    from conversations import get_conv_template
except ModuleNotFoundError:
    from .conversations import get_conv_template

try:
    from constants import (
        IMG_CONTEXT_TOKEN,
        IMG_END_TOKEN,
        IMG_START_TOKEN,
        CLIP_MEAN,
        CLIP_STD,
        IMAGENET_MEAN,
        IMAGENET_STD,
        SIGLIP_MEAN,
        SIGLIP_STD,
    )
except ModuleNotFoundError:
    from .constants import (
        IMG_CONTEXT_TOKEN,
        IMG_END_TOKEN,
        IMG_START_TOKEN,
        CLIP_MEAN,
        CLIP_STD,
        IMAGENET_MEAN,
        IMAGENET_STD,
        SIGLIP_MEAN,
        SIGLIP_STD,
    )

def get_item(
    data_item,
    template_name,
    root,
    tokenizer,
    tcs_loader,
    ds_name,
    num_image_token,
    group_by_length=False,
    use_packed_ds=True,
    min_num_frame=8,
    max_num_frame=32,
    sampling_method='rand',
    dynamic_image_size=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    use_thumbnail=False,
    is_train=False,  # NOTE (Vishaal, 14/05): patched this to not do any augmentation during computing the token stats
    image_size=448,
    pad2square=False,
    normalize_type='imagenet',
) -> Dict[str, torch.Tensor]:

    assert num_image_token == 256, "By default, the number of tokens per image-tile should be 256, please check again!"

    if "image" in data_item and len(data_item["image"]) != 0:
        if type(data_item["image"]) == list:
            ret = multi_modal_multi_image_get_item(
                data_item,
                template_name,
                tokenizer,
                num_image_token,
                group_by_length,
                use_packed_ds,
                ds_name,
                tcs_loader,
                dynamic_image_size,
                min_dynamic_patch,
                max_dynamic_patch,
                use_thumbnail,
                is_train,
                root,
                image_size,
                pad2square,
                normalize_type,
            )
        else:
            ret = multi_modal_get_item(
                data_item,
                template_name,
                tokenizer,
                num_image_token,
                group_by_length,
                use_packed_ds,
                ds_name,
                tcs_loader,
                dynamic_image_size,
                min_dynamic_patch,
                max_dynamic_patch,
                use_thumbnail,
                is_train,
                root,
                image_size,
                pad2square,
                normalize_type,
            )
    elif (
        "video" in data_item
        and data_item["video"] is not None
        and data_item["video"] != ""
    ):
        ret = video_get_item(
            data_item,
            template_name,
            tokenizer,
            num_image_token,
            group_by_length,
            use_packed_ds,
            ds_name,
            tcs_loader,
            max_num_frame,
            min_num_frame,
            sampling_method,
            is_train,
            root,
            image_size,
            pad2square,
            normalize_type,
        )
    else:
        ret = pure_text_get_item(
            data_item,
            template_name,
            tokenizer,
            num_image_token,
            group_by_length,
            use_packed_ds,
            ds_name,
            is_train,
            min_dynamic_patch,
            use_thumbnail,
            image_size,
            pad2square,
            normalize_type,
        )
    return ret


def load_image(tcs_loader, image_path):
    # Load the image using tcs_loader if available, otherwise use PIL
    if tcs_loader is not None and "s3://" in image_path:
        return tcs_loader(image_path)
    return Image.open(image_path).convert("RGB")


def get_image_path(image_path, root):
    if image_path.startswith("s3://"):  # for ceph
        image_path = root + image_path
    else:  # for local image
        image_path = os.path.join(root, image_path)
    return image_path


def get_transform(is_train, image_size, pad2square=False, normalize_type="imagenet"):
    # Build transformation function
    transform = build_transform(
        is_train=is_train,
        input_size=image_size,
        pad2square=pad2square,
        normalize_type=normalize_type,
    )
    return transform


def multi_modal_get_item(
    data_item,
    template_name,
    tokenizer,
    num_image_token,
    group_by_length,
    use_packed_ds,
    ds_name,
    tcs_loader,
    dynamic_image_size,
    min_dynamic_patch,
    max_dynamic_patch,
    use_thumbnail,
    is_train,
    root,
    image_size,
    pad2square,
    normalize_type,
):
    # Build transformation function
    transform = get_transform(is_train, image_size, pad2square, normalize_type)

    # Ensure the first conversation contains an image placeholder
    if "<image>" not in data_item["conversations"][0]["value"]:
        data_item["conversations"][0]["value"] = (
            "<image>\n" + data_item["conversations"][0]["value"]
        )

    # Merge the image path
    image_path = get_image_path(data_item["image"], root)

    # Load the image using tcs_loader if available, otherwise use PIL
    # NOTE: (Vishaal, 14/05): sometimes image loading might not work well, in those cases, we just create a standard 224x224 image
    # so that the token counter would count it as a standard square image.
    try:
        image = load_image(tcs_loader, image_path)
    except Exception as e:
        print(f"There was an error in loading {image_path}, defaulting to an all-white image like in the text-case!")
        # Create a blank white image
        image = Image.new("RGB", (224, 224), (255, 255, 255))

    if (
        dynamic_image_size
    ):  # If dynamic image size is enabled, preprocess the image dynamically
        images = dynamic_preprocess(
            image,
            min_num=min_dynamic_patch,
            max_num=max_dynamic_patch,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
        )
    else:  # Otherwise, use the original image as a single patch
        images = [image]

    # Apply the transformation to each image and stack the results into a tensor
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)

    # Ensure that there is only one patch if dynamic image size is not enabled
    num_patches = pixel_values.size(0)
    if not dynamic_image_size:
        assert (
            num_patches == 1
        ), f"The number of patches should be 1, but got {num_patches}."

    # Select the appropriate preprocessing function based on the template name
    preprocess_function = preprocess_internvl2_5

    # Preprocess the conversations and generate the return dictionary
    ret = preprocess_function(
        template_name,
        [deepcopy(data_item["conversations"])],
        tokenizer,
        [num_image_token * num_patches],
        group_by_length=group_by_length,
        use_packed_ds=use_packed_ds,
        ds_name=ds_name,
    )

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    check_ = (ret["input_ids"][0] == image_end_token_id).sum()
    #### NOTE (Vishaal, 13/05): we are commenting this out, since this is fine for us for computing the token lengths!
    # assert (
    #     check_ == 1
    # ), f"image tokens are truncated, there are {check_} image-end-tokens, should have been 1, this dataset is {ds_name}"

    # Create the final return dictionary
    ret = dict(
        input_ids=ret["input_ids"][0],
        text_only_input_ids=ret["text_only_input_ids"][0],
        labels=ret["labels"][0],
        attention_mask=ret["attention_mask"][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        trainable_token_count=ret["trainable_token_count"],
    )
    return ret


def multi_modal_multi_image_get_item(
    data_item,
    template_name,
    tokenizer,
    num_image_token,
    group_by_length,
    use_packed_ds,
    ds_name,
    tcs_loader,
    dynamic_image_size,
    min_dynamic_patch,
    max_dynamic_patch,
    use_thumbnail,
    is_train,
    root,
    image_size,
    pad2square,
    normalize_type,
):
    # Build transformation function
    transform = get_transform(is_train, image_size, pad2square, normalize_type)

    images, num_tiles = [], []
    num_image = len(data_item["image"])
    for image_path in data_item["image"]:
        # Merge the image path
        image_path = get_image_path(image_path, root)
        # Load the image using tcs_loader if available, otherwise use PIL
        # NOTE: (Vishaal, 14/05): sometimes image loading might not work well, in those cases, we just create a standard 224x224 image
        # so that the token counter would count it as a standard square image.
        try:
            image = load_image(tcs_loader, image_path)
        except Exception as e:
            print(f"There was an error in loading {image_path}, defaulting to an all-white image like in the text-case!")
            # Create a blank white image
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        if (
            dynamic_image_size
        ):  # If dynamic image size is enabled, preprocess the image dynamically
            image = dynamic_preprocess(
                image,
                min_num=min_dynamic_patch,
                max_num=max(1, max_dynamic_patch // num_image),
                image_size=image_size,
                use_thumbnail=use_thumbnail,
            )
            images += image
            num_tiles.append(len(image))
        else:  # Otherwise, use the original image as a single patch
            images.append(image)
            num_tiles.append(1)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    num_patches = pixel_values.size(0)

    # Select the appropriate preprocessing function based on the template name
    preprocess_function = preprocess_internvl2_5

    # Preprocess the conversations and generate the return dictionary
    num_image_tokens = [num_image_token * num_tile for num_tile in num_tiles]
    ret = preprocess_function(
        template_name,
        [deepcopy(data_item["conversations"])],
        tokenizer,
        num_image_tokens,
        group_by_length=group_by_length,
        use_packed_ds=use_packed_ds,
        ds_name=ds_name,
        num_image=num_image,
    )

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    check_ = (ret["input_ids"][0] == image_end_token_id).sum()
    #### NOTE (Vishaal, 13/05): we are commenting this out, since this is fine for us for computing the token lengths!
    # assert (
    #     check_ == num_image
    # ), f"image tokens are truncated, there are {check_} image-end-tokens, should have been 1, this dataset is {ds_name}"

    # Create the final return dictionary
    ret = dict(
        input_ids=ret["input_ids"][0],
        text_only_input_ids=ret["text_only_input_ids"][0],
        labels=ret["labels"][0],
        attention_mask=ret["attention_mask"][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        trainable_token_count=ret["trainable_token_count"],
    )
    return ret


def video_get_item(
    data_item,
    template_name,
    tokenizer,
    num_image_token,
    group_by_length,
    use_packed_ds,
    ds_name,
    tcs_loader,
    max_num_frame,
    min_num_frame,
    sampling_method,
    is_train,
    root,
    image_size,
    pad2square,
    normalize_type,
):
    # Build transformation function
    transform = get_transform(is_train, image_size, pad2square, normalize_type)

    # Ensure the first conversation contains a video placeholder
    if "<video>" not in data_item["conversations"][0]["value"]:
        data_item["conversations"][0]["value"] = (
            "<video>\n" + data_item["conversations"][0]["value"]
        )

    # Get the video file path
    video_file = data_item["video"]
    video_path = os.path.join(root, video_file)

    # Load the video frames using tcs_loader
    # TODO: Load videos without using tcsloader.
    image_list = tcs_loader(
        video_path,
        image_type="video",
        max_num_frames=max_num_frame,
        min_num_frames=min_num_frame,
        sample=sampling_method,
        clip=data_item.get("clip", None),
    )

    # Generate special tokens for each video frame
    special_tokens = "\n".join(
        ["Frame-{}: <image>".format(i + 1) for i in range(len(image_list))]
    )
    data_item["conversations"][0]["value"] = data_item["conversations"][0][
        "value"
    ].replace("<video>\n", special_tokens + "\n")

    # Transform each frame image and stack them into a tensor
    pixel_values = [transform(image) for image in image_list]
    pixel_values = torch.stack(pixel_values)
    num_patches = pixel_values.size(0)

    # Select the appropriate preprocessing function based on the template name
    preprocess_function = preprocess_internvl2_5

    # Preprocess the conversations and generate the return dictionary
    num_image_tokens = [num_image_token] * num_patches
    ret = preprocess_function(
        template_name,
        [deepcopy(data_item["conversations"])],
        tokenizer,
        num_image_tokens,
        group_by_length=group_by_length,
        use_packed_ds=use_packed_ds,
        ds_name=ds_name,
        num_image=num_patches,
    )

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)

    # Create the final return dictionary
    ret = dict(
        input_ids=ret["input_ids"][0],
        text_only_input_ids=ret["text_only_input_ids"][0],
        labels=ret["labels"][0],
        attention_mask=ret["attention_mask"][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        trainable_token_count=ret["trainable_token_count"],
    )
    return ret


def pure_text_get_item(
    data_item,
    template_name,
    tokenizer,
    num_image_token,
    group_by_length,
    use_packed_ds,
    ds_name,
    is_train,
    min_dynamic_patch,
    use_thumbnail,
    image_size,
    pad2square,
    normalize_type,
):
    # Build transformation function
    transform = get_transform(is_train, image_size, pad2square, normalize_type)

    # Create a blank white image
    image = Image.new("RGB", (224, 224), (255, 255, 255))

    # Dynamically preprocess the image to generate patches
    images = dynamic_preprocess(
        image,
        min_num=min_dynamic_patch,
        max_num=1,
        image_size=image_size,
        use_thumbnail=use_thumbnail,
    )

    # Apply the transformation to each image patch and stack them into a tensor
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    num_patches = pixel_values.size(0)

    # Ensure there is only one patch
    assert (
        num_patches == 1
    ), f"The number of patches should be 1, but got {num_patches}."

    # Select the appropriate preprocessing function based on the template name
    preprocess_function = preprocess_internvl2_5

    # Preprocess the conversations and generate the return dictionary
    ret = preprocess_function(
        template_name,
        [deepcopy(data_item["conversations"])],
        tokenizer,
        [num_image_token * num_patches],
        text_only=True,
        group_by_length=group_by_length,
        use_packed_ds=use_packed_ds,
        ds_name=ds_name,
    )

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)

    # Create the final return dictionary
    ret = dict(
        input_ids=ret["input_ids"][0],
        text_only_input_ids=ret["text_only_input_ids"][0],
        labels=ret["labels"][0],
        attention_mask=ret["attention_mask"][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        trainable_token_count=ret["trainable_token_count"],
    )
    return ret


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(
                output
            ).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {
    quality: simulate_jpeg_degradation(quality) for quality in qualities
}


def build_transform(is_train, input_size, pad2square=False, normalize_type="imagenet"):
    if normalize_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.RandomChoice(
                    [T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]
                ),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Lambda(
                        lambda img: expand2square(
                            img, tuple(int(x * 255) for x in MEAN)
                        )
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )

    return transform

#### NOTE (Vishaal, 14/05: patching this function to also return text-only tokens)
def preprocess_internvl2_5(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    assert len(sources) == 1, "process only the first conversations"
    conversations = sources[0]

    if conversations[0]["from"] == "system":
        system_prompt = conversations[0]["value"]
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
        # system_prompt = None

    text_only_convs = deepcopy(conversations)

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation["from"] == "human":
                image_cnt = conversation["value"].count("<image>")
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}"
                    conversation["value"] = conversation["value"].replace(
                        "<image>", image_tokens, 1
                    )
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f"{current_image_idx} != {num_image}"

    assert len(conversations) == len(text_only_convs)

    batches, roles = [], []
    text_only_batches = []
    if system_prompt is not None:
        batches.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        text_only_batches.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        roles.append("system")
    for conversation, tonly_conversation in zip(conversations, text_only_convs):
        if conversation["from"] == "human":
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            text_only_batches.append(f'<|im_start|>user\n{tonly_conversation["value"]}<|im_end|>\n')
            roles.append("human")
        elif conversation["from"] == "gpt":
            batches.append(
                f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n'
            )
            text_only_batches.append(
                f'<|im_start|>assistant\n{tonly_conversation["value"]}<|im_end|>\n'
            )
            roles.append("gpt")
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, "add_bos_token", False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]
        text_only_batches[0] = tokenizer.bos_token + text_only_batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors="np",
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    text_only_input_ids = tokenizer(
        text_only_batches,
        return_tensors="np",
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]
        text_only_input_ids = [item[1:] for item in text_only_input_ids]

    final_input_ids, final_text_only_input_ids, final_targets = [], [], []
    ignore_ids = tokenizer("<|im_start|>assistant\n", return_tensors="np").input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id, tonly_input_id in zip(roles, input_ids, text_only_input_ids):
        final_input_ids.append(input_id)
        final_text_only_input_ids.append(tonly_input_id)
        if role == "system" or role == "human":
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == "gpt":
            target = input_id.copy()
            target[:ignore_len] = (
                IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            )
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[
        : tokenizer.model_max_length
    ]
    text_only_input_ids = torch.tensor(np.concatenate(final_text_only_input_ids))[
        : tokenizer.model_max_length
    ]
    targets = torch.tensor(np.concatenate(final_targets))[: tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        text_only_input_ids = F.pad(text_only_input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    trainable_token_count = (targets != IGNORE_TOKEN_ID).sum().item()
    input_ids = input_ids.unsqueeze(0)
    text_only_input_ids = text_only_input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        text_only_input_ids=text_only_input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        trainable_token_count=trainable_token_count,
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    """
    This function first computes all the possible aspect ratios based on the min and max number of tiles allowed.

    Then, it would extract the closest aspect ratio from the set of possible ARs based on `find_closest_aspect_ratio`

    The total number of tiles in the end is equal to the Height * Width of the selected aspect ratio for the image.
    For example, if the closest target aspect ratio is (16, 9) [this is guaranteed to be less than max_num because of the selection], the image would be divided into 16 * 9 tiles.

    Note that the thumbnail is not accounted for in the max_num. It is only added later (see line 875)

    Finally, each tile is resized to (img_size, img_size) later in the function that calls this function.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
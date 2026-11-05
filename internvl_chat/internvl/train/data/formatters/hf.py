import torch
from copy import deepcopy

from internvl.train.dataset import (
    preprocess, preprocess_internlm,
    preprocess_internvl2_5, preprocess_mpt,
    preprocess_phi3
)
from internvl.train.constants import IMG_END_TOKEN


def get_preprocess_function(template_name: str) -> callable:
    # Select the appropriate preprocessing function based on the template name
    if template_name == 'Hermes-2':
        preprocess_function = preprocess_mpt
    elif template_name == 'internlm2-chat':
        preprocess_function = preprocess_internlm
    elif template_name == 'phi3-chat':
        preprocess_function = preprocess_phi3
    elif template_name == 'internvl2_5':
        preprocess_function = preprocess_internvl2_5
    else:
        preprocess_function = preprocess
    return preprocess_function


def huggingface_formatter(
    sample: dict,
    template_name: str,
    tokenizer,
    use_packed_ds: bool,
    max_tokens: int,
    persist_private_keys: bool = False
) -> dict:
    # detect whether the current is a text-only sample
    text_only = bool(sample['num_image'] == 0)

    preprocess_function = get_preprocess_function(template_name)
    ds_name = sample['__ds_name__']

    # Select the appropriate preprocessing function based on the template name
    ret = preprocess_function(
        template_name=template_name, 
        sources=[deepcopy(sample['conversations'])],
        tokenizer=tokenizer, 
        num_image_token_list=sample['num_image_tokens_per_image'],
        group_by_length=False,
        use_packed_ds=use_packed_ds,
        ds_name=ds_name, 
        num_image=sample['num_image'],
        text_only=text_only
    )

    # if we hit the token limit, we raise
    num_tokens = len(ret['input_ids'][0])
    if num_tokens > max_tokens:
        raise ValueError(f"Input exceeds max_tokens: {num_tokens} > {max_tokens}")

    # grab needed vars from the sample, then delete it 
    # to free up memory
    pixel_values = sample['pixel_values']
    num_img = sample['num_image']
    tot_num_tiles = sample['total_num_tiles']
    if persist_private_keys:
        pks = {k: v for k, v in sample.items() if k.startswith('_')}
    del sample

    # Calculate position_ids for packed dataset
    position_ids = ret['attention_mask'].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

    if not text_only:
        image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        check_ = (ret['input_ids'][0] == image_end_token_id).sum()
        error_msg = f"image tokens are truncated, "
        error_msg += f"there are {check_} image-end-tokens, should have been {num_img}"
        assert check_ == num_img, error_msg
        image_flags = torch.tensor([1] * tot_num_tiles, dtype=torch.long)
    else:
        image_flags = torch.tensor([0] * tot_num_tiles, dtype=torch.long)
    
    # Create the final return dictionary
    ret = dict(
        input_ids=ret['input_ids'][0],
        labels=ret['labels'][0],
        attention_mask=ret['attention_mask'][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=image_flags
    )
    if persist_private_keys:
        ret.update(pks)
    return ret
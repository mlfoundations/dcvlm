# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math

import torch
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from internvl.model.internvl_chat import (
    InternVLChatModel, 
    InternVLChatConfig, 
    InternVisionConfig, 
    InternVisionModel,
)
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.train.constants import (
    BOX_END_TOKEN, BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
    IMG_START_TOKEN, QUAD_END_TOKEN,
    QUAD_START_TOKEN, REF_END_TOKEN,
    REF_START_TOKEN
)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig


def split_model(num_layers, vit_alpha=0.5):
    device_map = {}
    world_size = torch.cuda.device_count()
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    device_map['language_model.model.rotary_emb'] = 0

    return device_map


def tie_lmhead(model):
    # manual tying cuz this doesn't seem to be working
    input_embeds = model.language_model.get_input_embeddings()
    output_embeds = model.language_model.get_output_embeddings()
    if not torch.allclose(input_embeds.weight, output_embeds.weight):
        print("Tying input and output embeddings...")
        output_embeds.weight = input_embeds.weight

    assert torch.allclose(
        model.get_input_embeddings().weight, 
        model.get_output_embeddings().weight
    )
    return model


def load_model_and_tokenizer(
    checkpoint: str, 
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    auto: bool = False
):
    import os
    import os.path as osp
    
    random_init = (
        "1B-Init" in checkpoint
        or "2B-Init" in checkpoint
        or "4B-Init" in checkpoint
    )
    if random_init:
        local_root = os.environ.get("OFFLINE_MODEL_DIR")
        vision_path = osp.join(local_root, "InternViT-300M-448px-V2_5")
        instruct = "Instruct-Init" in checkpoint
        suffix = "-Instruct" if instruct else ""
        if "InternLM2_5-Init" in checkpoint:
            if "2B-Init" not in checkpoint:
                raise ValueError(f"InternLM2_5-Init only supported for 2B size, got: {checkpoint}")
            llm_path = osp.join(local_root, "internlm2_5-1_8b-chat")
        elif "1B-Init" in checkpoint:
            llm_path = osp.join(local_root, f"Qwen2.5-0.5B{suffix}")
        elif "2B-Init" in checkpoint:
            llm_path = osp.join(local_root, f"Qwen2.5-1.5B{suffix}")
        elif "4B-Init" in checkpoint:
            llm_path = osp.join(local_root, f"Qwen2.5-3B{suffix}")
        return initialize_random_internvl2_5(vision_path, llm_path)

    # NOTE: (Matteo, 03/10), refactor this shared init fn
    # to always use the AutoModel API (tested with correct repro ofc)
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, 
        trust_remote_code=True, 
        use_fast=False
    )
    
    kwargs = {'trust_remote_code': True}
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)

    # define device map to split the model across GPUs
    if auto:
        num_hidden_layers = config.llm_config.num_hidden_layers
        device_map = split_model(num_hidden_layers)
        kwargs.update({'device_map': device_map})
    else:
        device_map = 'cpu'
        kwargs.update({'device_map': device_map})
    
    model = AutoModel.from_pretrained(
        checkpoint, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit, 
        load_in_4bit=load_in_4bit, 
        **kwargs
    ).eval()
    
    if not load_in_8bit and not load_in_4bit and not auto:
        model = model.cuda()
    
    if 'trust_remote_code' in kwargs:
        del kwargs['trust_remote_code']

    # NOTE: the HF Trainer removes the lm_head from dumped ckpts
    # cuz it assumes this is ok, but we get a warning when loading 
    # a ckpt, and the lm_head is randomly initialized; 
    # this might be due to a bug in how the InternVL config is crafted 
    # see comments in internvl_chat_pretrain.py 
    if config.llm_config.tie_word_embeddings:
        assert torch.allclose(
            model.language_model.get_input_embeddings().weight,
            model.language_model.get_output_embeddings().weight
        ), "Config says tie_word_embeddings=True, but the weights are different. Please check."
    
    return model, tokenizer



def initialize_random_internvl2_5(vision_path, llm_path):
    # NOTE: (Matteo, 14/05) this code is taken, without changes, 
    # from the internvl_chat_pretrain.py script. We are using this function
    # to initialize a random ckpt to get estimates for generation-based random chances; 
    # A caveat is that "internlm2_5-1_8b-chat", used to init InternVL2_5-2B, looks very bad!
    # which is inconsistent with the results of InternVL-1B. However, swapping in Qwen2.5-1.5B works very well
    print(f'Loading Vision-Encoder from {vision_path}')
    vision_config = InternVisionConfig.from_pretrained(vision_path)
    vision_model = InternVisionModel.from_pretrained(
        vision_path, 
        torch_dtype=torch.bfloat16, 
        config=vision_config
    )
    
    print(f'Loading LLM from {llm_path}')
    llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
    if llm_config.model_type == 'internlm2':
        model_type = InternLM2ForCausalLM
        llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        print('Using flash_attention_2 for InternLM')
    else:
        model_type = AutoModelForCausalLM
        llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        # print('Using flash_attention_2 for LLaMA')
    
    # NOTE: up to here, the token_emb matrix and the lm_head are tied 
    # if we're using the Qwen2.5 series
    llm = model_type.from_pretrained(
        llm_path, 
        torch_dtype=torch.bfloat16,
        config=llm_config, 
        trust_remote_code=True
    )

    # then in this config they manually set the following:
    # - self.tie_word_embeddings = False
    # - self.llm_config.tie_word_embeddings = self.tie_word_embeddings
    print('Building InternVLChatConfig...')
    internvl_chat_config = InternVLChatConfig(
        vision_config.to_dict(), 
        llm_config.to_dict(), 
        template="internvl2_5",
        dynamic_image_size=True,
        use_thumbnail=True, 
        ps_version='v2',
        min_dynamic_patch=1, 
        max_dynamic_patch=12
    )
        
    # and here they create the model, but since the model is initialized from PretrainedModel 
    # instances, it will just not use the updated config with tie_word_embeddings = False
    print('Building InternVLChatModel...')
    model = InternVLChatModel(
        internvl_chat_config, 
        vision_model, 
        llm,
        use_flash_attn=True
    )

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = llm_path
    print(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, 
        add_eos_token=False, 
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = 8192
    token_list = [
        IMG_START_TOKEN, 
        IMG_END_TOKEN, 
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN, 
        QUAD_END_TOKEN, 
        REF_START_TOKEN,
        REF_END_TOKEN, 
        BOX_START_TOKEN, 
        BOX_END_TOKEN
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        ### TODO: maybe we want to experiment with paligemma init?
        ### paligemma (https://arxiv.org/abs/2407.07726) fig 6 advocates for gaussian noise init
        ### currently, the AvgEmb option is advocated in https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        ### actually, maybe the AvgEmb init is already happening by default, see this PR: https://github.com/huggingface/transformers/pull/33325
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    
    model = model.eval().cuda().bfloat16()
    return model, tokenizer
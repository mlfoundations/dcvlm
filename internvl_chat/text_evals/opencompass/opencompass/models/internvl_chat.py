import os
import os.path as osp
import sys
import torch
import torch.distributed as dist
import warnings

# make sure the internvl_chat folder is added to PATH
current_dir = osp.dirname(osp.abspath(__file__))
while current_dir and osp.basename(current_dir) != 'internvl_chat':
    current_dir = osp.dirname(current_dir)
sys.path.append(current_dir)

# model functions
from internvl.model import load_model_and_tokenizer
from internvl.model.internvl_chat import InternVLChatModel
from .base import BaseModel, LMTemplateParser
from typing import Dict, List, Optional


class InternVLChat(BaseModel):

    def __init__(self, 
                 path: str, 
                 max_seq_len: int = 16384, 
                 tokenizer_only: bool = False, 
                 tokenizer_path: Optional[str] = None,
                 meta_template: Optional[Dict] = None,
                 use_mpo_prompt: bool = False):

        # detect if we're loading from a local ckpt
        self.model_path = path
        self.is_dmt_model = osp.isdir(self.model_path)
        if self.is_dmt_model: 
            warnings.warn(f"DMT Model detected! Loading from {self.model_path}")

        # setup prompt
        self.use_mpo_prompt = use_mpo_prompt
        self.use_cot = (os.getenv('USE_COT') == '1')

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        # Model and tokenizer init 
        self.model, self.tokenizer = load_model_and_tokenizer(
            checkpoint=self.model_path,
            load_in_4bit=False,
            load_in_8bit=False,
            auto=False # NOTE: we might need to change this for xlarge models
        )

        # Set device (for now we only distribute on 1 node at most)
        if dist.is_initialized():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = f"cuda:{local_rank}"
        else:
            self.device = "cuda"
        
        self.image_size = self.model.config.vision_config.image_size
        
        kwargs_default = dict(
            do_sample=False, 
            max_new_tokens=256, 
            top_p=None, 
            temperature=0.0
        )

        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    # we need to implement this fn for opencompass compat
    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        output = self.tokenizer([prompt], truncation=False)
        length = len(output['input_ids'][0])
        return length

    # we also need this fn for opencompass compat
    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:

        # text-only datasets with OpenCompass
        pixel_values = None
        num_patches_list = [0 for _ in range(len(inputs))]
        gen_cfg = dict(do_sample=False, max_new_tokens=max_out_len)

        # naive loop through messages
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            questions=inputs,
            generation_config=gen_cfg,
            verbose=True
        )

        return responses

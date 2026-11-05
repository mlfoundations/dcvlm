from mmengine.config import read_base
from opencompass.models import InternVLChat


with read_base():
    {{DATASET_IMPORT}}

max_tokens = 256
datasets = [*{{DATASET_NAMES}}]
model_path = {{MODEL_PATH}}
abbr = {{ABBR}}

models = [
    dict(
        type=InternVLChat,
        # Parameters for `HuggingFaceCausalLM` initialization.
        path=model_path,
        tokenizer_path=model_path,
        # NOTE use no padding and eval with bs=1
        # tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        # batch_padding=False,
        batch_size=16,              # The size of a batch during inference.
        max_seq_len=16384,
        max_out_len=max_tokens,            # Maximum number of generated tokens.
        # Common parameters shared by various models, not specific to `HuggingFaceCausalLM` initialization.
        abbr=abbr,            # Model abbreviation used for result display.
        run_cfg=dict(num_gpus=1),   # Run configuration to specify resource requirements.
    )
]

print(
    f"Created configuration for {len(models)} InternVL models with various max_new_tokens values"
)
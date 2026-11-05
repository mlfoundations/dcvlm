import os
import wandb
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from internvl.patch import make_transformers_compatible

class CheckpointCallback(TrainerCallback):
    """
    Callback to make Trainer checkpoints HF Compatible.
    """
    def patch_checkpoint(self, state, checkpoint):
        if (
            checkpoint is not None
            and state.is_world_process_zero
        ):
            print(f"Found existing checkpoint {checkpoint}, making HF compatible.")
            make_transformers_compatible(checkpoint)
    
    def on_save(self, args, state, control, **kwargs):
        checkpoint = get_last_checkpoint(args.output_dir)
        self.patch_checkpoint(state, checkpoint)

class ThroughputLogCallback(TrainerCallback):
    """
    Callback to log:
    - tokens/s
    - examples/s

    every :log_freq: steps
    """

    def on_log(self, args, state, control, **kwargs):
        model = kwargs['model']
        # W&B runs are only initialized on local-rank 0 procs
        if state.is_local_process_zero:
            wandb.log({
                "avg_pack_size": model.avg_fw_tokens,
                "train/tokens_per_second": model.toks_per_second,
                "train/samples_per_second": model.samples_per_second,
            })

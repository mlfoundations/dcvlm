# defines the training configurations for the DCVLM benchmark

SCALE_CONFIGS = {
    "debug": {
        # model stuff
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        # training stuff
        "batch_size": 64,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 50e6,
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "debug10x": {
        # model stuff
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        # training stuff
        "batch_size": 64,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 500e6,
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "small": {
        # model stuff
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        # training stuff
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 6.25 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
        # some other info
        # - num_nodes is typically 8
        # - tot_gpus is typically 32
        # - grad_acc steps become 16
    },
    "small_2bmodel": {
        "model_size": "2B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-1.5B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 6.25 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "small_4bmodel": {
        "model_size": "4B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-3B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 6.25 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "small2x": {
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 12.5 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "small2x_2bmodel": {
        "model_size": "2B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-1.5B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 12.5 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "small2x_4bmodel": {
        "model_size": "4B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-3B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 12.5 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "medium": {
        # model stuff
        "model_size": "2B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-1.5B",
        # training stuff
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 25 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
        # some other info
        # - num_nodes is typically 16
        # - tot_gpus is typically 64
        # - grad_acc steps become 8
    },
    "medium_1bmodel": {
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 25 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "medium_4bmodel": {
        "model_size": "4B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-3B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 25 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    # NOTE (14/08): not using the large scale so far
    # we might need to switch to 2D parallelism here
    # to avoid convergence issues with larger batch sizes
    "large": {
        # model stuff
        "model_size": "4B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-3B",
        # training stuff
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 100 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
        # some other info
        # - num_nodes is typically 64
        # - tot_gpus is typically 256
        # - grad_acc steps become 8
    },
    "large_1bmodel": {
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 50 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "large2x_1bmodel": {
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 100 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
        # typical: 64 nodes / 256 GPUs / GA=4
    },
    "xlarge_1bmodel": {
        "model_size": "1B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-0.5B",
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 200 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
        # some other info
        # - num_nodes is typically 64
        # - tot_gpus is typically 256
        # - grad_acc steps become 4
    },
    # 8B model at the 200B-token x-large budget. Mirrors "large" exactly
    # (batch_size=1024, lr=2e-5, all packing/warmup hparams identical); only the
    # model size (8B = InternViT-300M + Qwen2.5-7B) and token budget (200B) change.
    # NOTE: the previous xlarge entry (7B LM / batch_size=4096) was a stale,
    # Jupiter-only artifact that was never wired into launcher.sh; superseded here.
    "xlarge": {
        # model stuff
        "model_size": "8B",
        "vision_model": "InternViT-300M-448px-V2_5",
        "language_model": "Qwen2.5-7B",
        # training stuff
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 200 * (10**9),
        "max_packed_tokens": 8192,
        "num_images_expected": 24
        # some other info
        # - num_nodes is typically 128
        # - tot_gpus is typically 512
        # - grad_acc steps become 2
    },
    # ---- SFT scale configs ----
    # SFT token budget = 0.29 * pretraining token budget
    # 6.25B PT -> 1.8125B SFT, 12.5B PT -> 3.625B SFT, 25B PT -> 7.25B SFT, 100B PT -> 29B SFT
    # Model-size agnostic: MODEL_NAME_OR_PATH loads the full checkpoint directly
    "sft_1_8b": {
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 0.29 * 6.25 * (10**9),  # 1.8125B
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "sft_3_6b": {
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 0.29 * 12.5 * (10**9),  # 3.625B
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "sft_7_2b": {
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 0.29 * 25 * (10**9),  # 7.25B
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
    "sft_29b": {
        "batch_size": 1024,
        "lr": 2e-5,
        "warmup_ratio": 0.03,
        "tot_tokens": 0.29 * 100 * (10**9),  # 29B
        "max_packed_tokens": 8192,
        "num_images_expected": 24
    },
}
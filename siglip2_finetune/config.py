"""Hyperparameters for the SigLIP2 LoRA fine-tune (finetune_siglip2_lora.py).

Edit values here rather than in the training script.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FinetuneConfig:
    # Distinguishes this run's output dir (weights/fine-tuning/siglip2_lora_adapter_<run_name>/)
    # so separate experiments don't silently overwrite each other. Change this
    # before every run you want to keep and compare.
    run_name: str = "run2_bs32_ep15_cosine"

    ckpt: str = "google/siglip2-base-patch16-224"

    batch_size: int = 32
    epochs: int = 15
    seed: int = 0
    max_text_length: int = 64

    # Single shared lr for both LoRA and the heads - this is run2, the winning
    # config. (Differential lr was tried in two directions - LoRA 1e-4/head 2e-5,
    # and the reverse, LoRA 1e-5/head 1e-4 - both underperformed this.)
    lora_lr: float = 1e-4
    head_lr: float = 1e-4

    # LR schedule stepped once per training iteration (not per epoch): linear
    # warmup for the first `warmup_ratio` fraction of total steps, then cosine
    # decay to 0 over the rest.
    warmup_ratio: float = 0.1

    # LoRA adapters on all four attention projections, in both the vision and
    # text towers (peft matches module names by suffix).
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "out_proj")

    # Fully trainable (not LoRA) - the vision and text projection heads.
    modules_to_save: tuple = ("vision_model.head", "text_model.head")


CONFIG = FinetuneConfig()

"""Simple SigLIP2 fine-tune: LoRA on attention Q/K/V/O, full fine-tune on both
projection heads (vision_model.head = image side "VLM" head, text_model.head =
text side "LLM" head).

peft freezes every base-model weight except what a LoraConfig names, so:
  - target_modules=["q_proj", "k_proj", "v_proj", "out_proj"] -> LoRA adapters
    (small, low-rank) on all four attention projections in *both* towers (peft
    matches by suffix, so this hits vision_model.encoder.layers.*.self_attn and
    text_model.encoder.layers.*.self_attn alike).
  - modules_to_save=[...head]             -> these two modules stay fully
    trainable (not LoRA) and get saved whole, exactly as asked.
Everything else (MLPs, embeddings, logit_scale/bias) stays frozen.

Two learning rates, one per param group: LoRA's lora_B is zero-initialized (so
it contributes nothing at step 0 and needs a higher lr to move), while the
heads are already-pretrained weights getting fully fine-tuned rather than
reinitialized - a high lr there risks wrecking what they already learned. So
lora_lr > head_lr (see config.py), each following its own copy of the same
warmup+cosine shape, scaled from its own peak.

Training data: GLAMI-1M-dresses-pattern-length/manifest.json - 4125 dress photos
with English `description` captions (brand/color/garment translated from the
original ee/lt/lv/si/es product name, pattern_label/length_label stated
explicitly in the sentence), fine-tuned with SigLIP2's own sigmoid contrastive
loss (model(..., return_loss=True)). Trains on the manifest's "train" split
(3572 items) and reports before/after retrieval metrics on the held-out "test"
split (553 items), so the recall numbers reflect generalization, not memorization.
"""

import math
import random

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import get_cosine_schedule_with_warmup

from common import FINETUNE_DIR, IMG_DIR, load_items, load_siglip2, retrieval_metrics
from config import CONFIG

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = CONFIG.batch_size

LORA_CONFIG = LoraConfig(
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    bias="none",
    target_modules=list(CONFIG.lora_target_modules),
    modules_to_save=list(CONFIG.modules_to_save),
)


@torch.no_grad()
def embed_all(model, processor, items):
    model.eval()
    image_embeds, text_embeds = [], []
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        images = [Image.open(IMG_DIR / it["file"]).convert("RGB") for it in batch]
        caps = [it["description"] for it in batch]

        img_in = processor(images=images, return_tensors="pt").to(DEVICE)
        image_embeds.append(model.get_image_features(**img_in).float().cpu())

        txt_in = processor(
            text=caps, padding="max_length", max_length=CONFIG.max_text_length, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        text_embeds.append(model.get_text_features(**txt_in).float().cpu())
    image_embeds = torch.cat(image_embeds).numpy()
    text_embeds = torch.cat(text_embeds).numpy()
    image_embeds /= (image_embeds ** 2).sum(-1, keepdims=True) ** 0.5
    text_embeds /= (text_embeds ** 2).sum(-1, keepdims=True) ** 0.5
    return retrieval_metrics(image_embeds, text_embeds)


def main():
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)

    train_items = load_items(split="train")
    test_items = load_items(split="test")
    print(f"{len(train_items)} train / {len(test_items)} test image-caption pairs from manifest.json")

    print(f"Loading {CONFIG.ckpt} on {DEVICE} ...")
    base_model, processor = load_siglip2()
    base_model = base_model.to(DEVICE)

    print("\nBefore fine-tuning (held-out test split):")
    before = embed_all(base_model, processor, test_items)
    print(before)

    model = get_peft_model(base_model, LORA_CONFIG)
    model.print_trainable_parameters()

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "modules_to_save" in n]
    assert len(lora_params) + len(head_params) == sum(p.requires_grad for p in model.parameters()), (
        "trainable params outside lora_*/modules_to_save* - update the name filters above"
    )
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": CONFIG.lora_lr},
        {"params": head_params, "lr": CONFIG.head_lr},
    ])

    steps_per_epoch = math.ceil(len(train_items) / BATCH_SIZE)
    total_steps = steps_per_epoch * CONFIG.epochs
    warmup_steps = round(total_steps * CONFIG.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"LR schedule: {warmup_steps} warmup steps -> cosine decay over {total_steps} total steps "
          f"({steps_per_epoch} steps/epoch), peak lora_lr={CONFIG.lora_lr:.2e} head_lr={CONFIG.head_lr:.2e}")

    model.train()
    for epoch in range(CONFIG.epochs):
        order = list(range(len(train_items)))
        random.shuffle(order)
        epoch_loss, num_batches = 0.0, 0

        for i in range(0, len(order), BATCH_SIZE):
            batch = [train_items[j] for j in order[i : i + BATCH_SIZE]]
            images = [Image.open(IMG_DIR / it["file"]).convert("RGB") for it in batch]
            caps = [it["description"] for it in batch]

            img_in = processor(images=images, return_tensors="pt")
            txt_in = processor(
                text=caps, padding="max_length", max_length=CONFIG.max_text_length, truncation=True,
                return_tensors="pt",
            )
            inputs = {**img_in, **txt_in}
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # stepped per iteration, not per epoch

            epoch_loss += loss.item()
            num_batches += 1

        lora_lr, head_lr = scheduler.get_last_lr()
        print(f"epoch {epoch + 1}/{CONFIG.epochs}: loss={epoch_loss / num_batches:.4f} "
              f"lora_lr={lora_lr:.2e} head_lr={head_lr:.2e}")

    print("\nAfter fine-tuning (held-out test split):")
    after = embed_all(model, processor, test_items)
    print(after)

    out_dir = FINETUNE_DIR / f"siglip2_lora_adapter_{CONFIG.run_name}"
    model.save_pretrained(str(out_dir))
    print(f"\nSaved LoRA adapter + fine-tuned heads to {out_dir}")
    print("before:", before)
    print("after: ", after)


if __name__ == "__main__":
    main()

"""Shared helpers for SigLIP2 fine-tuning."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "datasets" / "GLAMI-1M-dresses-pattern-length"
MANIFEST = IMG_DIR / "manifest.json"

SIGLIP2_CKPT = "google/siglip2-base-patch16-224"
SIGLIP2_DIR = ROOT / "weights" / "siglip2"
FINETUNE_DIR = ROOT / "weights" / "fine-tuning"


def load_siglip2():
    """Download (once) into weights/siglip2/ instead of the default HF cache, and
    load from that local copy."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoProcessor

    if not (SIGLIP2_DIR / "model.safetensors").exists():
        print(f"Downloading {SIGLIP2_CKPT} to {SIGLIP2_DIR} (one-time) ...")
        snapshot_download(repo_id=SIGLIP2_CKPT, local_dir=str(SIGLIP2_DIR))

    model = AutoModel.from_pretrained(str(SIGLIP2_DIR))
    processor = AutoProcessor.from_pretrained(str(SIGLIP2_DIR))
    return model, processor


def load_items(split=None):
    """Return list of {file, description, ...} dicts, sorted by filename.

    Pass split="train" or split="test" to filter by the manifest's own split
    field instead of loading everything.
    """
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = data["items"]
    if split is not None:
        items = [it for it in items if it["split"] == split]
    return sorted(items, key=lambda it: it["file"])


def retrieval_ranks(image_embeds: np.ndarray, text_embeds: np.ndarray):
    """Per-item 0-indexed rank of the correct match, both directions (i2t, t2i).

    Row/col i's positive is caption/image i (same item), scored against every
    other item in the batch - so ranks[i] == 0 means item i's own image (resp.
    caption) was the top-1 nearest neighbor among all of `image_embeds` (resp.
    `text_embeds`).
    """
    sims = image_embeds @ text_embeds.T  # [N, N], row i = image i vs all captions
    n = sims.shape[0]
    labels = np.arange(n)

    def _ranks(scores):
        order = np.argsort(-scores, axis=1)
        ranks = np.zeros(n, dtype=int)
        for i in range(n):
            ranks[i] = int(np.where(order[i] == labels[i])[0][0])
        return ranks

    return _ranks(sims), _ranks(sims.T)


def summarize_ranks(ranks: np.ndarray):
    return {
        "recall@1": float(np.mean(ranks == 0)),
        "recall@5": float(np.mean(ranks < 5)),
        "mrr": float(np.mean(1.0 / (ranks + 1))),
    }


def retrieval_metrics(image_embeds: np.ndarray, text_embeds: np.ndarray):
    """Image<->text retrieval eval with each image's own caption as the positive.

    Returns recall@1/@5 and mean reciprocal rank, both directions (i2t, t2i).
    """
    ranks_i2t, ranks_t2i = retrieval_ranks(image_embeds, text_embeds)
    return {
        "image_to_text": summarize_ranks(ranks_i2t),
        "text_to_image": summarize_ranks(ranks_t2i),
    }

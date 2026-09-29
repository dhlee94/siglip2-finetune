"""Materialize the pattern+length text-derived subset as a standalone dataset
folder, mirroring how build_glami_dresses.py built GLAMI-1M-dresses/:

Output: ../datasets/GLAMI-1M-dresses-pattern-length/{images/, manifest.json}

Reads results/pattern_length_dataset.jsonl (from build_dataset.py) and copies just
those items' images out of GLAMI-1M-dresses/images/.
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DATASET_DIR = ROOT / "datasets" / "GLAMI-1M-dresses"
LABELS_PATH = Path(__file__).resolve().parent / "results" / "pattern_length_dataset.jsonl"

OUT_DIR = ROOT / "datasets" / "GLAMI-1M-dresses-pattern-length"
OUT_IMAGES = OUT_DIR / "images"


def main():
    rows = [json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    items = []
    missing = 0
    for row in rows:
        src_img = SRC_DATASET_DIR / row["file"]
        if not src_img.exists():
            missing += 1
            continue
        dst_img = OUT_IMAGES / Path(row["file"]).name
        shutil.copy2(src_img, dst_img)
        items.append({
            "item_id": row["item_id"],
            "split": row["split"],
            "geo": row["geo"],
            "file": f"images/{Path(row['file']).name}",
            "name": row["name"],
            "pattern_label": row["pattern_label"],
            "length_label": row["length_label"],
        })

    manifest = {
        "role": "GLAMI-1M-dresses-pattern-length",
        "source": "GLAMI-1M-dresses, text-derived pattern+length labels "
                   "(parsed from description field for geos ee/lt/lv/si/es; "
                   "see build_glami_dataset/)",
        "count": len(items),
        "missing_images": missing,
        "items": items,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Copied {len(items)} images to {OUT_IMAGES} ({missing} missing).")
    print(f"Manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

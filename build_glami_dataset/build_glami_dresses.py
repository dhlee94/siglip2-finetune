"""Filter the GLAMI-1M dataset down to just the "dresses" category and copy those
images + metadata into a standalone folder in this project.

Source: ~/Downloads/GLAMI-1M-dataset/{GLAMI-1M-train.csv,GLAMI-1M-test.csv,images/}
Output: ./datasets/GLAMI-1M-dresses/{images/, manifest.json}
"""

import csv
import json
import shutil
from pathlib import Path

SRC_ROOT = Path.home() / "Downloads" / "GLAMI-1M-dataset"
SRC_IMAGES = SRC_ROOT / "images"
SRC_CSVS = {"train": SRC_ROOT / "GLAMI-1M-train.csv", "test": SRC_ROOT / "GLAMI-1M-test.csv"}

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "GLAMI-1M-dresses"
OUT_IMAGES = OUT_DIR / "images"

CATEGORY_NAME = "dresses"  # excludes girls-dresses, men/women-dress-shoes


def main():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    items = []
    missing = 0
    for split, csv_path in SRC_CSVS.items():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["category_name"] != CATEGORY_NAME:
                    continue
                src_img = SRC_IMAGES / f"{row['image_id']}.jpg"
                if not src_img.exists():
                    missing += 1
                    continue
                dst_name = f"{row['image_id']}.jpg"
                shutil.copy2(src_img, OUT_IMAGES / dst_name)
                items.append(
                    {
                        "split": split,
                        "item_id": row["item_id"],
                        "image_id": row["image_id"],
                        "file": f"images/{dst_name}",
                        "geo": row["geo"],
                        "name": row["name"],
                        "description": row["description"],
                        "category": row["category"],
                        "category_name": row["category_name"],
                        "label_source": row["label_source"],
                    }
                )
                if len(items) % 2000 == 0:
                    print(f"  copied {len(items)} so far...")

    manifest = {
        "role": "GLAMI-1M-dresses",
        "source": "GLAMI-1M-dataset (category_name == 'dresses')",
        "count": len(items),
        "missing_images": missing,
        "items": items,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. {len(items)} dress images copied to {OUT_IMAGES} ({missing} referenced files were missing).")
    print(f"Manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

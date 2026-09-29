"""Build a manifest.json for pexels-dresses-crops/ (100 Pexels dress photos, cropped
to the dress region) with the same pattern_label/length_label/description shape
as GLAMI-1M-dresses-pattern-length, so the two datasets are directly comparable.

Unlike GLAMI, this source has no structured "Pattern:/Length:" fields to parse -
its pexels-dresses/manifest.json `alt` captions are free-form scene descriptions
that almost never state hem length (checked all 100: only one item explicitly
says "maxi dress"; the rest have zero length cues in text). So pattern_label and
length_label here are NOT parsed from data - they are Claude's own visual
judgment from looking at each of the 100 crop images directly (see
pexels_labels.py), using body proportion to estimate length whenever the hem
itself fell outside the crop. Treat these as AI-assisted pseudo-labels, not
verified ground truth like GLAMI's.

Usage: python3 build_pexels_crops_manifest.py
"""

import json
from pathlib import Path

from pexels_labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = ROOT / "datasets" / "pexels-dresses-crops"
ORIG_MANIFEST = ROOT / "datasets" / "pexels-dresses" / "manifest.json"

PATTERN_PHRASE = {
    "solid": "a solid, single-color",
    "floral": "a floral-print",
    "striped": "a striped",
    "animal": "an animal-print",
    "polka_dot": "a polka-dot",
    "mixed": "a mixed-color / heathered",
    "checkered": "a checkered",
    "logo": "a logo-print",
}

LENGTH_PHRASE = {
    "mini": "mini-length (short)",
    "three_quarter": "three-quarter length",
    "knee_midi": "knee-length (midi)",
    "seven_eighth": "seven-eighths length",
    "maxi": "maxi-length (floor-length)",
}

OUT_PATH = CROPS_DIR / "manifest.json"


def main():
    orig = json.loads(ORIG_MANIFEST.read_text(encoding="utf-8"))
    orig_by_file = {it["file"]: it for it in orig["items"]}

    missing = [f for f in orig_by_file if f not in LABELS]
    if missing:
        raise ValueError(f"no pattern/length label for: {missing}")

    items = []
    for file, (pattern_label, length_label) in LABELS.items():
        alt = orig_by_file[file]["alt"].rstrip()
        alt_sentence = alt if alt.endswith((".", "!", "?")) else alt + "."
        description = (
            f"{alt_sentence} It has {PATTERN_PHRASE[pattern_label]} pattern and is {LENGTH_PHRASE[length_label]}."
        )
        items.append({
            "file": file,
            "alt": alt,
            "pattern_label": pattern_label,
            "length_label": length_label,
            "description": description,
        })
    items.sort(key=lambda it: it["file"])

    manifest = {
        "role": "pexels-dresses-crops-pattern-length",
        "source": "pexels-dresses/manifest.json (100 Pexels dress photos, cropped to the "
                   "dress region). pattern_label/length_label are Claude's visual "
                   "judgment from viewing each crop image directly - NOT parsed from "
                   "structured data like GLAMI's, since these alt-text captions "
                   "essentially never state hem length. Treat as AI-assisted "
                   "pseudo-labels, not verified ground truth. See build_pexels_crops_manifest.py.",
        "count": len(items),
        "items": items,
    }
    OUT_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()

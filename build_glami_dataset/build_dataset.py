"""Build the final text-derived training subset: for each GLAMI-1M-dresses item in
geos ee/lt/lv/si/es, extract pattern/length from `description` (extract.py) and
normalize to shared English labels (normalize.py). An item is kept only if at
least one of pattern_label / length_label was actually found in its text - nothing
is filled in for a missing field.

Output: results/pattern_length_dataset.jsonl
  Each record's "file" is relative to ../datasets/GLAMI-1M-dresses/, i.e. the same
  images already on disk there - no image copying needed.
"""

import json
from collections import Counter
from pathlib import Path

from extract import SCHEMAS, split_fields, extract_pattern, extract_length
from normalize import normalize_pattern, normalize_length

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets" / "GLAMI-1M-dresses" / "manifest.json"
OUT_DIR = Path(__file__).resolve().parent / "results"

# Classes below this many examples (counted over the full deduped set, before this
# cutoff was applied) aren't worth keeping - too few to train or evaluate on.
# Drops pattern: two_tone(38) paisley(37) ethnic(29) graphic(26) gradient(18)
# moto_print(18) oriental(11) graphic_text(4) camo(1); length: half_length(7).
MIN_CLASS_COUNT = 50
EXCLUDED_PATTERN_LABELS = {
    "two_tone", "paisley", "ethnic", "graphic", "gradient",
    "moto_print", "oriental", "graphic_text", "camo",
}
EXCLUDED_LENGTH_LABELS = {"half_length"}


def main():
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_files = set()
    dup_images_skipped = 0
    pattern_counts = Counter()
    length_counts = Counter()
    dropped_unmapped_pattern = Counter()
    dropped_unmapped_length = Counter()

    for it in items:
        geo = it["geo"]
        schema = SCHEMAS.get(geo)
        if schema is None:
            continue

        desc = it.get("description") or ""
        segments = split_fields(desc, schema)
        pattern_raw = extract_pattern(segments, schema)
        length_raw = extract_length(segments, schema)

        pattern_label = normalize_pattern(geo, pattern_raw)
        length_label = normalize_length(geo, length_raw)

        if pattern_raw is not None and pattern_label is None:
            dropped_unmapped_pattern[(geo, pattern_raw)] += 1
        if length_raw is not None and length_label is None:
            dropped_unmapped_length[(geo, length_raw)] += 1

        if pattern_label is None or length_label is None:
            continue  # keep only items with BOTH a pattern and a length label

        if pattern_label in EXCLUDED_PATTERN_LABELS or length_label in EXCLUDED_LENGTH_LABELS:
            continue  # too few examples of this class to be usable (see MIN_CLASS_COUNT)

        if it["file"] in seen_files:
            # same product photo re-listed under another geo's storefront with an
            # identical label - keep only the first listing so it isn't overweighted
            dup_images_skipped += 1
            continue
        seen_files.add(it["file"])

        if pattern_label:
            pattern_counts[pattern_label] += 1
        if length_label:
            length_counts[length_label] += 1

        rows.append({
            "item_id": it["item_id"],
            "split": it["split"],
            "geo": geo,
            "file": it["file"],
            "name": it["name"],
            "pattern_label": pattern_label,
            "length_label": length_label,
        })

    out_path = OUT_DIR / "pattern_length_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pattern = sum(1 for r in rows if r["pattern_label"])
    n_length = sum(1 for r in rows if r["length_label"])
    n_both = sum(1 for r in rows if r["pattern_label"] and r["length_label"])

    print(f"Duplicate-image rows skipped (same photo re-listed under another geo): {dup_images_skipped}")
    print(f"Total rows kept: {len(rows)}")
    print(f"  with pattern_label: {n_pattern}")
    print(f"  with length_label:  {n_length}")
    print(f"  with both:          {n_both}")

    print("\npattern_label distribution:")
    for label, c in pattern_counts.most_common():
        print(f"  {label:15s} {c}")

    print("\nlength_label distribution:")
    for label, c in length_counts.most_common():
        print(f"  {label:15s} {c}")

    if dropped_unmapped_pattern:
        print(f"\nDropped (pattern found but not mapped to a canonical label): "
              f"{sum(dropped_unmapped_pattern.values())} items")
        for (geo, raw), c in dropped_unmapped_pattern.most_common(20):
            print(f"  [{geo}] {c:3d}  {raw!r}")

    if dropped_unmapped_length:
        print(f"\nDropped (length found but not mapped to a canonical label): "
              f"{sum(dropped_unmapped_length.values())} items")
        for (geo, raw), c in dropped_unmapped_length.most_common(20):
            print(f"  [{geo}] {c:3d}  {raw!r}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

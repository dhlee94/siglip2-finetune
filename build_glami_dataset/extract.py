"""Extract "pattern" (패턴) and "length" (기장) labels for GLAMI-1M-dresses items,
using ONLY text already present in each item's `description` (the structured
"Label: value" template used by geos ee/lt/lv/si/es - see schema.py). No value is
guessed or filled in: if an item's description doesn't contain that field, the
item is simply left out of that attribute's output.

Usage:
    python3 extract.py                # writes results/extracted.jsonl + prints summary
"""

import json
import re
from collections import Counter
from pathlib import Path

from schema import SCHEMAS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets" / "GLAMI-1M-dresses" / "manifest.json"
OUT_DIR = Path(__file__).resolve().parent / "results"


def compile_boundary_re(schema):
    alternation = "|".join(re.escape(l) for l in schema.boundary_labels)
    return re.compile(rf"(?:{alternation}):\s*")


_BOUNDARY_RE_CACHE = {geo: compile_boundary_re(s) for geo, s in SCHEMAS.items()}


def split_fields(description, schema):
    """Return list of (label, value) segments found in `description`, using the
    locale's known field-label whitelist to find segment boundaries."""
    boundary_re = _BOUNDARY_RE_CACHE[schema.geo]
    label_re = re.compile(
        "(" + "|".join(re.escape(l) for l in schema.boundary_labels) + "):\\s*"
    )
    matches = list(label_re.finditer(description))
    segments = []
    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(description)
        value = description[start:end].strip().rstrip(",").strip()
        if value:
            segments.append((label, value))
    return segments


def extract_pattern(segments, schema):
    values = [v for lbl, v in segments if lbl == schema.pattern_label]
    return values[0] if values else None


def extract_length(segments, schema):
    values = [v for lbl, v in segments if lbl == schema.length_label]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    filtered = [
        v for v in values
        if not any(marker.lower() in v.lower() for marker in schema.length_cut_markers)
    ]
    if filtered:
        return filtered[0]
    return values[-1]


def main():
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    stats = Counter()
    pattern_values = {geo: Counter() for geo in SCHEMAS}
    length_values = {geo: Counter() for geo in SCHEMAS}

    for it in items:
        geo = it["geo"]
        schema = SCHEMAS.get(geo)
        if schema is None:
            continue  # geo without a known structured template - skip, don't guess
        stats["in_scope_geo"] += 1

        desc = it.get("description") or ""
        segments = split_fields(desc, schema)

        pattern_raw = extract_pattern(segments, schema)
        length_raw = extract_length(segments, schema)

        if pattern_raw is None and length_raw is None:
            continue  # nothing usable in this item's text - drop it, don't fabricate

        if pattern_raw is not None:
            stats["pattern_found"] += 1
            pattern_values[geo][pattern_raw] += 1
        if length_raw is not None:
            stats["length_found"] += 1
            length_values[geo][length_raw] += 1
        if pattern_raw is not None and length_raw is not None:
            stats["both_found"] += 1

        results.append({
            "item_id": it["item_id"],
            "geo": geo,
            "file": it["file"],
            "name": it["name"],
            "pattern_raw": pattern_raw,
            "length_raw": length_raw,
        })

    out_path = OUT_DIR / "extracted.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total dataset items: {len(items)}")
    print(f"In-scope geo (ee/lt/lv/si/es) items: {stats['in_scope_geo']}")
    print(f"Items with >=1 usable field: {len(results)}")
    print(f"  pattern_raw found: {stats['pattern_found']}")
    print(f"  length_raw found:  {stats['length_found']}")
    print(f"  both found:        {stats['both_found']}")
    print(f"Saved: {out_path}")

    for geo in SCHEMAS:
        print(f"\n=== geo={geo} distinct pattern_raw values (top 20) ===")
        for v, c in pattern_values[geo].most_common(20):
            print(f"  {c:5d}  {v!r}")
        print(f"--- geo={geo} distinct length_raw values (top 20) ---")
        for v, c in length_values[geo].most_common(20):
            print(f"  {c:5d}  {v!r}")


if __name__ == "__main__":
    main()

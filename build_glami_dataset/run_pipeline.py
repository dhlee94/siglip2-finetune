"""Run the full GLAMI-1M -> dresses -> pattern/length dataset pipeline, start to
finish, in one command:

  1. build_glami_dresses.py        raw GLAMI-1M (needs ~/Downloads/GLAMI-1M-dataset/,
                                    the original dataset download - not included in
                                    this repo) -> datasets/GLAMI-1M-dresses/
  2. build_dataset.py              parse structured description text + normalize to
                                    canonical English pattern/length labels
                                    -> results/pattern_length_dataset.jsonl
  3. export_dataset.py             copy just the labeled images + write the final
                                    manifest -> datasets/GLAMI-1M-dresses-pattern-length/
  4. add_english_descriptions.py   add the English caption `description` field
                                    used as the fine-tuning/eval caption, in place

extract.py is intentionally NOT part of this chain - it's a standalone diagnostic
dump (results/extracted.jsonl + printed per-locale value counts) that was used to
hand-curate normalize.py's translation dictionaries in the first place; nothing
downstream reads its output file. Run it directly if you want to see those raw
distinct values again.

Each step is skipped if its output already exists, so this is safe to re-run or
resume from a partial run. Pass --force to redo every step regardless.

Usage:
    python3 run_pipeline.py
    python3 run_pipeline.py --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# (script, output file that marks the step as already done - None means "always run")
STEPS = [
    ("build_glami_dresses.py", ROOT / "datasets" / "GLAMI-1M-dresses" / "manifest.json"),
    ("build_dataset.py", HERE / "results" / "pattern_length_dataset.jsonl"),
    ("export_dataset.py", ROOT / "datasets" / "GLAMI-1M-dresses-pattern-length" / "manifest.json"),
    ("add_english_descriptions.py", None),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-run every step even if its output already exists")
    args = parser.parse_args()

    for script, output_marker in STEPS:
        if output_marker is not None and output_marker.exists() and not args.force:
            print(f"[skip] {script}  (output already exists: {output_marker})")
            continue
        print(f"[run]  {script}")
        subprocess.run([sys.executable, script], cwd=HERE, check=True)

    print("\nDone. Final dataset: datasets/GLAMI-1M-dresses-pattern-length/")


if __name__ == "__main__":
    main()

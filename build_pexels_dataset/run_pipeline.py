"""Run the full pexels-dresses -> crops -> pattern/length caption pipeline in one
command:

  1. crop_dress.py                    detect + crop each photo to its dress
                                       bounding box -> datasets/pexels-dresses-crops/
  2. build_pexels_crops_manifest.py   add Claude-judged pattern_label/length_label
                                       + the English `description` caption ->
                                       datasets/pexels-dresses-crops/manifest.json

detect_dress_face.py is intentionally NOT part of this chain - it's a standalone
QA tool used to visually sanity-check the dress/face detectors before writing
crop_dress.py, not something the pipeline depends on. Run it directly if you
want to re-inspect detection quality.

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
CROPS_DIR = ROOT / "datasets" / "pexels-dresses-crops"

# (script, "is this step already done?" check) - a plain directory's existence
# isn't a reliable marker here since crop_dress.py mkdir's it before populating
# it, so step 1 checks for an actual cropped image instead.
STEPS = [
    ("crop_dress.py", lambda: any(CROPS_DIR.glob("*.jpg"))),
    ("build_pexels_crops_manifest.py", lambda: (CROPS_DIR / "manifest.json").exists()),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-run every step even if its output already exists")
    args = parser.parse_args()

    for script, already_done in STEPS:
        if already_done() and not args.force:
            print(f"[skip] {script}  (output already exists)")
            continue
        print(f"[run]  {script}")
        subprocess.run([sys.executable, script], cwd=HERE, check=True)

    print("\nDone. Final dataset: datasets/pexels-dresses-crops/")


if __name__ == "__main__":
    main()

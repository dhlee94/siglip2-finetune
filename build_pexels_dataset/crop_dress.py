"""Crop each pexels-dresses photo to its dress bounding box (no face exclusion).

Same YOLO-World dress detection as detect_dress_face.py (single highest-confidence
box per image), but instead of drawing the box on the full image, the image is
cropped to that box and saved - and the face-exclusion step is skipped entirely
(detect_dress_face.py's face-aware box adjustment was a QA/visualization check on
the detector, not something this - the actual pipeline script - carries over).
The box is padded by MARGIN_RATIO on each side (clamped to image bounds) so the
crop isn't cut flush against the garment.

Reads from and writes to the shared ../datasets/ folder (not duplicated here) -
this folder holds only code + the weights it depends on.

Output: ../datasets/pexels-dresses-crops/<original filename>.jpg
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT.parent / "datasets"
SRC_DIR = DATASETS / "pexels-dresses"
OUT_DIR = DATASETS / "pexels-dresses-crops"

DRESS_WEIGHTS = ROOT / "weights" / "yolov8s-worldv2.pt"

CLASSES = ["dress"]
DRESS_CONF = 0.05
DEVICE = "mps"
MARGIN_RATIO = 0.15  # pad each side by 15% of the box's width/height


def best_dress_box(model, img_path):
    results = model.predict(source=str(img_path), conf=DRESS_CONF, device=DEVICE, verbose=False)
    r = results[0]
    if len(r.boxes) == 0:
        return None
    confs = r.boxes.conf.tolist()
    xyxy = r.boxes.xyxy.tolist()
    best_idx = max(range(len(confs)), key=lambda i: confs[i])
    x1, y1, x2, y2 = xyxy[best_idx]
    return (x1, y1, x2, y2, confs[best_idx])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(DRESS_WEIGHTS))
    model.set_classes(CLASSES)

    images = sorted(SRC_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images")

    ok, no_dress = 0, 0
    for i, img_path in enumerate(images, 1):
        box = best_dress_box(model, img_path)
        if box is None:
            no_dress += 1
            print(f"[{i}/{len(images)}] {img_path.name}: no-dress")
            continue

        x1, y1, x2, y2, conf = box
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        mx = (x2 - x1) * MARGIN_RATIO
        my = (y2 - y1) * MARGIN_RATIO
        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))

        crop = img[y1:y2, x1:x2]
        cv2.imwrite(str(OUT_DIR / img_path.name), crop)
        ok += 1
        print(f"[{i}/{len(images)}] {img_path.name}: cropped (conf={conf:.2f})")

    print(f"\nDone. {ok}/{len(images)} crops saved to {OUT_DIR} ({no_dress} skipped, no dress detected)")


if __name__ == "__main__":
    main()

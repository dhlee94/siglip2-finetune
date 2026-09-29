"""Detect a dress on each person photo and box it, excluding any overlapping face.

- Dress detection: YOLO-World v2 (open-vocabulary), class = "dress".
  Only the single highest-confidence box per image is kept.
- Face detection: MediaPipe FaceDetection, full-range model (BlazeFace variant
  tuned for faces far from the camera, as in full-body photos). Tried OpenCV's
  YuNet first, but it threw frequent false positives on busy backgrounds and
  even on dress patterns (e.g. polka dots got detected as tiny faces); the
  MediaPipe full-range model had zero false positives across a 30-image
  sample while still finding the real face in most shots.
- If a detected face overlaps the dress box, the dress box's top edge is
  pushed down past the face's bottom edge so the face is excluded.
  A face is only treated as an overlap if it starts above the dress box's
  top edge (real anatomy: face sits above the garment) - this filters out
  any remaining stray "face" detections that land inside the dress box.

Draws both boxes on the full image (dress + face), unlike crop_dress.py which
crops to the dress box alone with no face exclusion. Reads from and writes to
the shared ../datasets/ folder (not duplicated here) - this folder holds only
code + the weights it depends on.

Standalone QA tool, NOT part of the reproducible dataset pipeline: this is how
the YOLO-World dress detector + MediaPipe face detector combo was visually
sanity-checked before writing crop_dress.py. Its output (pexels-dresses-dress-boxes/,
annotated full images) is a one-off debug artifact, not consumed by anything
downstream - crop_dress.py is the actual pipeline script.
"""

from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT.parent / "datasets"
SRC_DIR = DATASETS / "pexels-dresses"
OUT_DIR = DATASETS / "pexels-dresses-dress-boxes"

DRESS_WEIGHTS = ROOT / "weights" / "yolov8s-worldv2.pt"

CLASSES = ["dress"]
DRESS_CONF = 0.05  # low threshold; we pick the single best box ourselves
DEVICE = "mps"  # use "cpu" if no Apple GPU / CUDA available

FACE_MODEL_SELECTION = 1  # 0 = short-range (<2m), 1 = full-range (better for full-body shots)
FACE_MIN_CONF = 0.5


def best_dress_box(model, img_path):
    """Return (x1, y1, x2, y2, conf) for the highest-confidence dress box, or None."""
    results = model.predict(source=str(img_path), conf=DRESS_CONF, device=DEVICE, verbose=False)
    r = results[0]
    if len(r.boxes) == 0:
        return None
    confs = r.boxes.conf.tolist()
    xyxy = r.boxes.xyxy.tolist()
    best_idx = max(range(len(confs)), key=lambda i: confs[i])
    x1, y1, x2, y2 = xyxy[best_idx]
    return (x1, y1, x2, y2, confs[best_idx])


def detect_faces(face_detector, image_bgr):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = face_detector.process(image_rgb)
    boxes = []
    for det in result.detections or []:
        score = det.score[0]
        box = det.location_data.relative_bounding_box
        x1, y1 = box.xmin * w, box.ymin * h
        x2, y2 = x1 + box.width * w, y1 + box.height * h
        boxes.append((x1, y1, x2, y2, float(score)))
    return boxes


def exclude_faces_from_dress(dress_box, face_boxes):
    """Clip the top of the dress box down past any overlapping face's bottom edge."""
    dx1, dy1, dx2, dy2, conf = dress_box
    for fx1, fy1, fx2, fy2, _ in face_boxes:
        ix1, iy1 = max(fx1, dx1), max(fy1, dy1)
        ix2, iy2 = min(fx2, dx2), min(fy2, dy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue  # no intersection
        if fy1 >= dy1:
            # A real face sits above the garment. One that starts inside/below
            # the dress box top is a stray detection, not the wearer's face.
            continue
        dy1 = max(dy1, fy2)
    if dy1 >= dy2:
        return None  # face swallowed the whole box; drop it
    return (dx1, dy1, dx2, dy2, conf)


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dress_model = YOLO(str(DRESS_WEIGHTS))
    dress_model.set_classes(CLASSES)

    images = sorted(SRC_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images")

    no_dress, clipped = 0, 0
    with mp.solutions.face_detection.FaceDetection(
        model_selection=FACE_MODEL_SELECTION, min_detection_confidence=FACE_MIN_CONF
    ) as face_detector:
        for i, img_path in enumerate(images, 1):
            dress = best_dress_box(dress_model, img_path)
            img = cv2.imread(str(img_path))
            faces = detect_faces(face_detector, img)

            status = "no-dress"
            if dress is not None:
                adjusted = exclude_faces_from_dress(dress, faces)
                if adjusted is None:
                    no_dress += 1
                    status = "dress-removed(face-overlap)"
                else:
                    status = "dress(face-clipped)" if adjusted[1] != dress[1] else "dress"
                    if status == "dress(face-clipped)":
                        clipped += 1
                    draw_box(img, adjusted, (255, 128, 0), f"dress {adjusted[4]:.2f}")
            else:
                no_dress += 1

            for face in faces:
                draw_box(img, face, (0, 0, 255), f"face {face[4]:.2f}")

            cv2.imwrite(str(OUT_DIR / img_path.name), img)
            print(f"[{i}/{len(images)}] {img_path.name}: faces={len(faces)} -> {status}")

    kept = len(images) - no_dress
    print(f"\nDone. {kept}/{len(images)} images kept a dress box ({clipped} clipped for face overlap).")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

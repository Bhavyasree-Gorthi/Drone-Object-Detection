"""
AI-Based Drone Object Detection
--------------------------------
Simple pipeline:  Video/Image  ->  YOLOv8  ->  Detect objects  ->  Draw boxes  ->  Save/Show result

Detects (from COCO classes, filtered to the ones relevant to drone footage):
  person, car, bus, motorcycle, bicycle, truck, train

Usage:
    python detect.py --source path/to/video.mp4
    python detect.py --source path/to/image.jpg
    python detect.py --source 0                # webcam
    python detect.py --source path/to/video.mp4 --show   # also show live preview window
"""

import argparse
import os
import time

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Objects we care about for drone footage (subset of the 80 COCO classes).
# Set this to None to detect ALL 80 COCO classes instead.
TARGET_CLASSES = ["person", "car", "bus", "motorcycle", "bicycle", "truck", "train"]

CONFIDENCE_THRESHOLD = 0.35

# A distinct BGR color per class name, so boxes are easy to tell apart
COLORS = {
    "person": (0, 255, 0),
    "car": (255, 128, 0),
    "bus": (0, 128, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (0, 255, 255),
    "truck": (128, 0, 255),
    "train": (255, 255, 0),
}
DEFAULT_COLOR = (200, 200, 200)


def load_model():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    return model


def get_class_filter(model):
    """Convert TARGET_CLASSES names -> class-id list expected by ultralytics."""
    if TARGET_CLASSES is None:
        return None
    name_to_id = {v: k for k, v in model.names.items()}
    ids = [name_to_id[n] for n in TARGET_CLASSES if n in name_to_id]
    return ids


def draw_detections(frame, result):
    """Draw bounding boxes + labels on a frame given one ultralytics Result."""
    counts = {}
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        counts[label] = counts.get(label, 0) + 1

    return frame, counts


def draw_summary(frame, counts, fps=None):
    """Small summary panel in the top-left corner."""
    y = 25
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 25
    for label, count in counts.items():
        cv2.putText(frame, f"{label}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 25
    return frame


def run_on_image(model, class_ids, source, show=False):
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    results = model(frame, classes=class_ids, conf=CONFIDENCE_THRESHOLD, verbose=False)
    frame, counts = draw_detections(frame, results[0])
    frame = draw_summary(frame, counts)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "result_" + os.path.basename(source))
    cv2.imwrite(out_path, frame)
    print(f"Saved: {out_path}")
    print("Detections:", counts)

    if show:
        cv2.imshow("Drone Object Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(model, class_ids, source, show=False):
    # source can be a filepath (str) or webcam index (int)
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    is_webcam = isinstance(cap_source, int)
    out_name = "result_webcam.mp4" if is_webcam else "result_" + os.path.basename(str(source))
    out_path = os.path.join(OUTPUT_DIR, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), in_fps, (width, height))

    print("Processing... press 'q' in the preview window to stop early (if --show is used).")
    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=class_ids, conf=CONFIDENCE_THRESHOLD, verbose=False)
        frame, counts = draw_detections(frame, results[0])

        frame_count += 1
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        frame = draw_summary(frame, counts, fps=fps)

        writer.write(frame)

        if show:
            cv2.imshow("Drone Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Saved: {out_path}")
    print(f"Processed {frame_count} frames in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="AI-Based Drone Object Detection")
    parser.add_argument("--source", required=True,
                         help="Path to video/image file, or webcam index (e.g. 0)")
    parser.add_argument("--show", action="store_true",
                         help="Show a live preview window while processing")
    args = parser.parse_args()

    model = load_model()
    class_ids = get_class_filter(model)

    source = args.source
    is_webcam = source.isdigit()
    is_image = (not is_webcam) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        run_on_image(model, class_ids, source, show=args.show)
    else:
        run_on_video(model, class_ids, source, show=args.show)


if __name__ == "__main__":
    main()

"""
Streamlit UI for the Drone Object Detection project.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

from detect import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    get_class_filter,
    draw_detections,
    draw_summary,
)

st.set_page_config(page_title="Drone Object Detection", page_icon="🛰️", layout="centered")

st.title("🛰️ AI-Based Drone Object Detection")
st.write("Upload a drone image or video and let YOLO detect people, cars, buses, motorcycles, and bicycles.")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()
class_ids = get_class_filter(model)

st.sidebar.header("Detection settings")
confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.05,
    max_value=0.90,
    value=0.10,
    step=0.05,
    help="Lower values find smaller or less certain objects, but may add false detections.",
)
image_size = st.sidebar.select_slider(
    "Inference image size",
    options=[640, 960, 1280],
    value=1280,
    help="Larger sizes can improve detection of small objects and use more CPU time.",
)

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"],
)

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix in (".jpg", ".jpeg", ".png", ".webp"):
        frame = cv2.imread(tmp_path)
        results = model(frame, classes=class_ids, conf=confidence, imgsz=image_size, verbose=False)
        frame, counts = draw_detections(frame, results[0])
        frame = draw_summary(frame, counts)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
        st.write("### Detections")
        st.json(counts)

    else:
        st.write("Processing video, please wait...")
        cap = cv2.VideoCapture(tmp_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        out_path = tmp_path + "_result.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0
        total_counts = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, classes=class_ids, conf=confidence, imgsz=image_size, verbose=False)
            frame, counts = draw_detections(frame, results[0])
            frame = draw_summary(frame, counts)
            writer.write(frame)

            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v

            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        writer.release()

        st.success("Done!")
        st.video(out_path)
        st.write("### Total detections across video")
        st.json(total_counts)

st.sidebar.header("About")
st.sidebar.write(
    "Pipeline: Drone footage → YOLOv8 → Object detection → Bounding boxes → Result.\n\n"
    "Detected classes: person, car, bus, motorcycle, bicycle, truck, train."
)

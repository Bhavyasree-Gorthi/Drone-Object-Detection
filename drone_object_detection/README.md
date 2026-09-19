# AI-Based Drone Object Detection

A simple mini-project that runs object detection on drone video/image footage using YOLOv8.

**Pipeline:** Drone video/image → YOLO → Detect objects → Draw bounding boxes → Display/save result

Detects:
- 👤 Person
- 🚗 Car
- 🚌 Bus
- 🏍️ Motorcycle
- 🚲 Bicycle
- 🚚 Truck
- 🚆 Train

## Project Structure

```
drone_object_detection/
├── detect.py          # Main script: run detection on image/video/webcam
├── app.py              # Optional Streamlit web UI
├── yolov8n.pt           # Pre-trained YOLOv8 nano model (auto-downloaded, included here)
├── requirements.txt     # Python dependencies
├── sample_media/        # Put your own test images/videos here
├── outputs/              # Results are saved here
└── README.md
```

## 1. Setup

```bash
cd drone_object_detection
pip install -r requirements.txt
```

(If `yolov8n.pt` isn't already in the folder, it will auto-download the first time you run the script — no manual steps needed.)

## 2. Run on a video (command line)

```bash
python detect.py --source sample_media/your_drone_video.mp4
```

Add `--show` to also see a live preview window while it processes:

```bash
python detect.py --source sample_media/your_drone_video.mp4 --show
```

## 3. Run on an image

```bash
python detect.py --source sample_media/your_photo.jpg
```

## 4. Run on your webcam (quick test without drone footage)

```bash
python detect.py --source 0 --show
```

## 5. (Optional) Run the Streamlit web app

```bash
streamlit run app.py
```

This opens a browser page where you can upload an image or video and see the detections directly.

## Output

- Processed image/video files are saved to `outputs/`
- Each detected object gets a colored bounding box + label + confidence score
- A small on-screen summary shows live counts per object class (and FPS for video)

## How it works (short version)

1. Each frame is read with OpenCV.
2. The frame is passed to a pre-trained YOLOv8 model (`ultralytics` library), which
   returns bounding boxes, class labels, and confidence scores.
3. Only the classes we care about (person, car, bus, motorcycle, bicycle, truck, train)
   are kept — this is controlled by the `TARGET_CLASSES` list in `detect.py`.
4. Boxes and labels are drawn on the frame with OpenCV, and the frame is written to
   the output video (or saved as an image).

## Notes for your own drone footage

- No GPU is required — YOLOv8n (nano) runs fine on CPU for a demo/mini-project, just slower on long videos.
- If you don't have real drone footage, any top-down or aerial video (even from YouTube, downloaded locally) works fine as a test source — just make sure you have rights to use it.
- To detect ALL 80 COCO classes instead of just the 7 selected here, open `detect.py`
  and set `TARGET_CLASSES = None`.

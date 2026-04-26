import streamlit as st
import torch
import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="WeatherGuard", layout="wide")
st.title("WeatherGuard – Adaptive Object Detection")

# -------------------------------------------------------
# MODEL LOADING (CACHED)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

yolo_model = load_model()

# -------------------------------------------------------
# IMAGE RESTORATION PIPELINE
# -------------------------------------------------------
def anti_glare_restoration(img_pil: Image.Image, upsample: bool = False):
    img_np = np.array(img_pil)
    if img_np.size == 0:
        return img_pil, np.zeros((1, 1, 3), dtype=np.uint8)

    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

    if upsample:
        h, w = img_cv.shape[:2]
        img_cv = cv2.resize(img_cv, (int(w * 1.25), int(h * 1.25)), interpolation=cv2.INTER_CUBIC)

    clip_val = 0.95
    img_clipped = np.minimum(img_cv, clip_val) / clip_val
    img_clipped = np.clip(img_clipped, 0, 1)

    gamma = 0.9
    img_gamma = np.power(img_clipped, gamma)

    img_8u = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_8u, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    img_den = cv2.bilateralFilter(img_eq, d=7, sigmaColor=70, sigmaSpace=70)

    blur = cv2.GaussianBlur(img_den, (0, 0), sigmaX=1.2)
    img_sharp = cv2.addWeighted(img_den, 1.35, blur, -0.35, 0)

    final_pil = Image.fromarray(cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB))
    return final_pil, img_sharp

# -------------------------------------------------------
# ADAPTATION ENGINE
# -------------------------------------------------------
def adaptation_engine(condition: str):
    conf = 0.25
    min_bbox_area = 0

    if condition == "clear":
        conf = 0.7
        min_bbox_area = 50
    elif condition == "rainy":
        conf = 0.4
        min_bbox_area = 30
    elif condition == "foggy":
        conf = 0.35
        min_bbox_area = 70
    elif condition == "glare":
        conf = 0.6
        min_bbox_area = 100

    return conf, min_bbox_area

# -------------------------------------------------------
# OBJECT DETECTION
# -------------------------------------------------------
def supervised_object_detection(yolo_model, img_bgr, conf, min_bbox_area):
    results = yolo_model(img_bgr, conf=conf, verbose=False)
    output = img_bgr.copy()
    detections = []

    if results and results[0].boxes:
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            area = (x2 - x1) * (y2 - y1)

            if area < min_bbox_area:
                continue

            detections.append(box)

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = yolo_model.names[int(cls)]

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                f"{label} {score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    return output, detections

# -------------------------------------------------------
# VIDEO PROCESSING
# -------------------------------------------------------
def process_video(video_path, condition, show_stream):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video")
        return

    frame_placeholder = st.empty()
    processed = 0
    start = time.time()

    conf, min_area = adaptation_engine(condition)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, restored = anti_glare_restoration(pil)

        detected, _ = supervised_object_detection(
            yolo_model, restored, conf, min_area
        )

        if show_stream and processed % 3 == 0:
            rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, use_container_width=True)

        processed += 1

    cap.release()
    elapsed = time.time() - start
    st.success(f"Processed {processed} frames in {elapsed:.2f}s")

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

condition = st.selectbox(
    "Environmental Condition",
    ["clear", "rainy", "foggy", "glare"],
)

show_stream = st.checkbox("Live Preview", value=True)

if uploaded_video:
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    if st.button("Run Processing"):
        process_video("input_video.mp4", condition, show_stream)

!pip install -q streamlit ultralytics opencv-python pillow

%%writefile app.py

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="WeatherGuard",
    layout="wide"
)

st.title("🌦️ WeatherGuard – Adaptive Object Detection")
st.write(
    "Weather-aware object detection using image restoration "
    "and YOLOv8."
)

# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


yolo_model = load_model()

# -------------------------------------------------------
# IMAGE RESTORATION PIPELINE
# -------------------------------------------------------
def anti_glare_restoration(
    img_pil: Image.Image,
    upsample: bool = False
):

    img_np = np.array(img_pil)

    if img_np.size == 0:
        return img_pil, np.zeros(
            (1, 1, 3),
            dtype=np.uint8
        )

    # RGB -> BGR
    img_cv = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    ).astype(np.float32) / 255.0

    # ---------------------------------------------------
    # Optional Upsampling
    # ---------------------------------------------------
    if upsample:

        h, w = img_cv.shape[:2]

        img_cv = cv2.resize(
            img_cv,
            (
                int(w * 1.25),
                int(h * 1.25)
            ),
            interpolation=cv2.INTER_CUBIC
        )

    # ---------------------------------------------------
    # Highlight / Glare Suppression
    # ---------------------------------------------------
    clip_val = 0.95

    img_clipped = np.minimum(
        img_cv,
        clip_val
    ) / clip_val

    img_clipped = np.clip(
        img_clipped,
        0,
        1
    )

    # ---------------------------------------------------
    # Gamma Correction
    # ---------------------------------------------------
    gamma = 0.9

    img_gamma = np.power(
        img_clipped,
        gamma
    )

    img_8u = np.clip(
        img_gamma * 255,
        0,
        255
    ).astype(np.uint8)

    # ---------------------------------------------------
    # CLAHE
    # ---------------------------------------------------
    lab = cv2.cvtColor(
        img_8u,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l_eq = clahe.apply(l)

    lab_eq = cv2.merge(
        (l_eq, a, b)
    )

    img_eq = cv2.cvtColor(
        lab_eq,
        cv2.COLOR_LAB2BGR
    )

    # ---------------------------------------------------
    # Denoising
    # ---------------------------------------------------
    img_den = cv2.bilateralFilter(
        img_eq,
        d=7,
        sigmaColor=70,
        sigmaSpace=70
    )

    # ---------------------------------------------------
    # Sharpening
    # ---------------------------------------------------
    blur = cv2.GaussianBlur(
        img_den,
        (0, 0),
        sigmaX=1.2
    )

    img_sharp = cv2.addWeighted(
        img_den,
        1.35,
        blur,
        -0.35,
        0
    )

    # BGR -> RGB
    final_pil = Image.fromarray(
        cv2.cvtColor(
            img_sharp,
            cv2.COLOR_BGR2RGB
        )
    )

    return final_pil, img_sharp


# -------------------------------------------------------
# ADAPTATION ENGINE
# -------------------------------------------------------
def adaptation_engine(condition):

    conf = 0.25
    min_bbox_area = 0

    if condition == "clear":

        conf = 0.70
        min_bbox_area = 50

    elif condition == "rainy":

        conf = 0.40
        min_bbox_area = 30

    elif condition == "foggy":

        conf = 0.35
        min_bbox_area = 70

    elif condition == "glare":

        conf = 0.60
        min_bbox_area = 100

    return conf, min_bbox_area


# -------------------------------------------------------
# OBJECT DETECTION
# -------------------------------------------------------
def supervised_object_detection(
    model,
    img_bgr,
    conf,
    min_bbox_area
):

    results = model(
        img_bgr,
        conf=conf,
        verbose=False
    )

    output = img_bgr.copy()

    detections = []

    if results and results[0].boxes:

        for box in results[0].boxes.data.tolist():

            x1, y1, x2, y2, score, cls = box

            # Bounding box area
            area = (
                (x2 - x1) *
                (y2 - y1)
            )

            # Remove very small boxes
            if area < min_bbox_area:
                continue

            detections.append(box)

            x1, y1, x2, y2 = map(
                int,
                [x1, y1, x2, y2]
            )

            label = model.names[int(cls)]

            # Draw bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            cv2.putText(
                output,
                f"{label} {score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    return output, detections


# -------------------------------------------------------
# VIDEO PROCESSING
# -------------------------------------------------------
def process_video(
    video_path,
    condition,
    show_stream
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        st.error("Could not open video.")

        return None

    # Video properties
    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30

    # ---------------------------------------------------
    # Output video path
    # ---------------------------------------------------
    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    # Try mp4 codec
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # ---------------------------------------------------
    # UI placeholders
    # ---------------------------------------------------
    frame_placeholder = st.empty()

    progress_bar = st.progress(0)

    status_text = st.empty()

    # ---------------------------------------------------
    # Adaptation parameters
    # ---------------------------------------------------
    conf, min_area = adaptation_engine(
        condition
    )

    st.info(
        f"Condition: **{condition}** | "
        f"Confidence: **{conf}** | "
        f"Minimum Box Area: **{min_area}**"
    )

    processed = 0
    total_detections = 0

    start = time.time()

    # ---------------------------------------------------
    # Frame processing loop
    # ---------------------------------------------------
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # ------------------------------------------------
        # Convert frame to PIL
        # ------------------------------------------------
        pil = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        # ------------------------------------------------
        # Image restoration
        # ------------------------------------------------
        _, restored = anti_glare_restoration(
            pil
        )

        # ------------------------------------------------
        # YOLO detection
        # ------------------------------------------------
        detected, detections = (
            supervised_object_detection(
                yolo_model,
                restored,
                conf,
                min_area
            )
        )

        total_detections += len(
            detections
        )

        # ------------------------------------------------
        # Write frame to output video
        # ------------------------------------------------
        writer.write(detected)

        # ------------------------------------------------
        # Live preview
        # ------------------------------------------------
        if show_stream and processed % 3 == 0:

            rgb = cv2.cvtColor(
                detected,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                rgb,
                use_container_width=True
            )

        # ------------------------------------------------
        # Progress
        # ------------------------------------------------
        processed += 1

        if total_frames > 0:

            progress = (
                processed /
                total_frames
            )

            progress_bar.progress(
                min(progress, 1.0)
            )

        status_text.text(
            f"Processing frame "
            f"{processed}/{total_frames}"
        )

    # ---------------------------------------------------
    # Cleanup
    # ---------------------------------------------------
    cap.release()
    writer.release()

    elapsed = time.time() - start

    progress_bar.progress(1.0)

    status_text.success(
        f"Completed! {processed} frames processed "
        f"in {elapsed:.2f} seconds."
    )

    st.success(
        f"Total detections: {total_detections}"
    )

    return output_path


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

st.sidebar.header("⚙️ Settings")

condition = st.sidebar.selectbox(
    "Environmental Condition",
    [
        "clear",
        "rainy",
        "foggy",
        "glare"
    ]
)

show_stream = st.sidebar.checkbox(
    "Live Preview",
    value=True
)

st.subheader("📹 Upload Video")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "avi",
        "mov"
    ]
)

# -------------------------------------------------------
# Show selected condition
# -------------------------------------------------------
if condition == "clear":

    st.info(
        "☀️ Clear: Higher confidence threshold "
        "for reliable detections."
    )

elif condition == "rainy":

    st.info(
        "🌧️ Rainy: Lower confidence threshold "
        "to detect objects affected by rain."
    )

elif condition == "foggy":

    st.info(
        "🌫️ Foggy: Lower confidence threshold "
        "to compensate for reduced visibility."
    )

elif condition == "glare":

    st.info(
        "💡 Glare: Higher confidence and "
        "larger minimum bounding box."
    )


# -------------------------------------------------------
# Process uploaded video
# -------------------------------------------------------
if uploaded_video:

    st.video(
        uploaded_video
    )

    if st.button(
        "🚀 Run WeatherGuard",
        type="primary"
    ):

        # Save uploaded video
        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        # Process
        output_path = process_video(
            input_file.name,
            condition,
            show_stream
        )

        # ------------------------------------------------
        # Download output
        # ------------------------------------------------
        if output_path and os.path.exists(
            output_path
        ):

            st.subheader(
                "🎥 Processed Video"
            )

            st.video(
                output_path
            )

            with open(
                output_path,
                "rb"
            ) as video_file:

                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_file,
                    file_name="weatherguard_output.mp4",
                    mime="video/mp4"
                )




!pkill -9 -f streamlit

!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableXsrfProtection false --server.enableCORS false > /content/streamlit.log 2>&1 &


!cat /content/streamlit.log


!curl -I http://localhost:8501



from google.colab.output import eval_js

url = eval_js("google.colab.kernel.proxyPort(8501)")
print(url)


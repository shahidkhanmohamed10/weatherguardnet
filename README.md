# WeatherGuardNet

## About the Project

WeatherGuardNet is a computer vision project for detecting objects in video under different environmental conditions such as clear weather, rain, fog, and glare.

The project uses **YOLOv8** for object detection and **OpenCV** for image enhancement. Different detection settings are used depending on the weather condition selected by the user.

The application is built using **Streamlit**, so the user can upload a video, select the environmental condition, process the video, and download the result.

## What the Project Does

The application follows these main steps:

1. Upload a video.
2. Select the environmental condition.
3. Enhance the video frames using OpenCV.
4. Run YOLOv8 object detection.
5. Filter detections based on confidence and bounding-box size.
6. Display the processed video.
7. Download the processed video.

## Environmental Conditions

The application currently supports four conditions:

* Clear
* Rainy
* Foggy
* Glare

Each condition uses different image enhancement settings and detection confidence thresholds.

## Technologies Used

* Python
* YOLOv8
* Ultralytics
* OpenCV
* PyTorch
* NumPy
* Pillow
* Streamlit

## Model

The project uses the **YOLOv8 Nano (****`yolov8n.pt`****)** model for object detection.

YOLOv8 is used because it provides a good balance between detection speed and accuracy, making it suitable for video-based computer vision applications.

## Image Enhancement

Before object detection, the video frames are processed using OpenCV.

The enhancement pipeline includes techniques such as:

* Gamma correction
* CLAHE
* Bilateral filtering
* Sharpening
* Contrast enhancement

The purpose of these steps is to improve the visual quality of frames before they are passed to the object detection model.

## Project Structure

```text
WeatherGuardNet/
├── README.md
├── WhatsApp Video 2026-08-25 at 11.44.29 AM.mp4
├── app.py
└── requirements.txt
```

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/shahidkhanmohamed10/weatherguardnet.git
```

Go to the project folder:

```bash
cd weatherguardnet
```

### 2. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

### 4. Use the Application

* Upload a video file.
* Select the environmental condition.
* Start processing.
* View the detected objects in the output video.
* Download the processed video.

## Requirements

The main Python libraries required are:

* streamlit
* torch
* torchvision
* torchaudio
* ultralytics
* opencv-python-headless
* numpy
* Pillow

The project can run on CPU, although a GPU can provide faster video processing and object detection.

## Example Workflow

```text
Input Video
     ↓
Select Weather Condition
     ↓
OpenCV Image Enhancement
     ↓
YOLOv8 Object Detection
     ↓
Confidence & Bounding Box Filtering
     ↓
Processed Video
     ↓
Download Result
```

## Current Limitations

The weather condition is currently **selected by the user**. The system does not automatically classify whether a video is rainy, foggy, clear, or affected by glare.

The project also uses fixed enhancement parameters and confidence thresholds for each selected condition.

The current implementation has not been evaluated on a dedicated weather-specific object detection benchmark.

## Future Improvements

Some possible improvements are:

* Add automatic weather-condition classification.
* Train or fine-tune YOLOv8 using weather-specific datasets.
* Compare object detection performance before and after image enhancement.
* Test the system on larger and more varied video datasets.
* Improve processing speed for real-time applications.

## Author

**Shahid Khan Mohammed**

GitHub:
https://github.com/shahidkhanmohamed10

## Repository

https://github.com/shahidkhanmohamed10/weatherguardnet

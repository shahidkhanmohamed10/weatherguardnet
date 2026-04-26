WeatherGuardNet

WeatherGuardNet is a computer vision project focused on improving object detection in extreme weather conditions such as rain, fog, snow, sandstorms, and night glare.

In real-world environments, object detection models often perform poorly when visibility is low. WeatherGuardNet tries to solve this problem by combining diffusion-based image restoration, object detection, and an adaptive learning mechanism that allows the model to adjust to new weather conditions.

The main goal of the project is to improve detection accuracy even when weather conditions change or become severe.

Project Idea

Many object detection systems work well under normal lighting and weather conditions but struggle when the environment becomes challenging. Heavy fog, rain, or night glare can hide important objects in a scene and reduce model performance.

WeatherGuardNet addresses this issue by restoring degraded images before detection and allowing the system to adapt during inference. This helps the model maintain reliable performance even when it encounters weather conditions that were not present during training.

How the System Works

The system follows a simple pipeline:

An input image or video frame is provided to the system.

The model estimates the weather condition affecting the scene.

A diffusion-based model restores the degraded image.

Important visual features are extracted from the restored image.

An object detection model identifies objects and their locations.

A test-time adaptation module updates the model to improve performance in new weather conditions.

Pipeline Input Image ↓ Weather Detection ↓ Diffusion-based Image Restoration ↓ Feature Extraction ↓ Object Detection ↓ Output Detected Objects ↓ Model Adaptation for Next Frame Technologies Used

The implementation uses the following tools and technologies:

DDPM (Denoising Diffusion Probabilistic Model) – used for image restoration

Stable Diffusion – helps generate clearer reconstructed images

YOLO (You Only Look Once) – used for object detection

Test-Time Adaptation (TTA) – allows the model to adapt during inference

AdaBN (Adaptive Batch Normalization) – helps align features during adaptation

PyTorch – main deep learning framework used in development

CUDA – enables GPU acceleration for faster training and inference

Research Motivation

Existing vision models have several limitations when working in adverse weather environments:

Many models only handle one type of weather degradation

Image restoration and object detection are often treated as separate tasks

Most systems cannot adapt to new weather conditions in real time

WeatherGuardNet attempts to solve these problems by combining restoration, detection, and adaptation into one unified system.

Expected Outcome

With this approach, the system aims to:

Improve image clarity in difficult weather conditions

Detect objects more reliably in degraded environments

Maintain stable performance across different weather scenarios

Adapt to unseen weather conditions without retraining

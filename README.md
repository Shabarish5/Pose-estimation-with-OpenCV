# Pose-estimation-with-OpenCV

This repository contains the streamlit_base.py application for real-time pose estimation using OpenCV's deep learning module (cv.dnn).

Features

Pose Estimation on Images, Videos, and Live Webcam

Real-time Processing

Download Processed Videos

Streamlit UI for Easy Interaction

Requirements

Ensure you have the following dependencies installed:

pip install streamlit opencv-python numpy ffmpeg-python

Running the Application

To run the OpenCV-based pose estimation app:

streamlit run streamlit_base.py

Model Files Required

Ensure that graph_opt.pb (TensorFlow pose model) is in the same directory before running the application.

Usage

Upload an image or video for processing.

Select live webcam to process real-time pose estimation.

Download processed video after analysis.

License

This project is open-source under the MIT License.

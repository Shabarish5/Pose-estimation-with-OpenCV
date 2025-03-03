import streamlit as st
import cv2 as cv
import numpy as np
import tempfile
import os
from datetime import datetime
import ffmpeg

st.title("Pose Estimation using OpenCV")

# Load the pre-trained model
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Define body parts and pose pairs
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
              ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
              ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
              ["Nose", "LEye"], ["LEye", "LEar"]]

# Function to process an image/video frame
def process_frame(frame):
    inWidth, inHeight = 368, 368
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.09 else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    return frame

# Input selection
option = st.radio("Select Input Type:", ("Image", "Video", "Live Webcam"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv.imdecode(file_bytes, 1)
        output_frame = process_frame(frame)
        st.image(output_frame, channels="BGR")
        st.success("Pose estimation completed for the image.")

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        cap = cv.VideoCapture(tfile.name)
        out_file = "output_video.mp4"
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(temp_output, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()

        # Convert to MP4
        ffmpeg.input(temp_output).output(out_file, vcodec='libx264', format='mp4').run()
        os.remove(temp_output)

        st.video(out_file)
        st.download_button("Download Processed Video", out_file, file_name="processed_video.mp4")
        st.success("Pose estimation completed for the video.")

elif option == "Live Webcam":
    record = st.checkbox("Record Session")
    temp_video = "recorded_video.avi"
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(temp_video, fourcc, 20.0, (640, 480))

    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        if record:
            out.write(processed_frame)
        stframe.image(processed_frame, channels="BGR")
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()

    mp4_output = "recorded_video.mp4"
    ffmpeg.input(temp_video).output(mp4_output, vcodec='libx264', format='mp4').run()
    os.remove(temp_video)

    st.success("Recording Completed! Pose estimation completed for the live webcam session.")
    st.download_button("Download Recorded Video", mp4_output, file_name=f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

import math
import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(
    page_title="Drowsiness Detection System", page_icon="🚗", layout="centered"
)

st.title("🚗 ระบบตรวจจับการหลับใน (Drowsiness Detection)")
st.write("อัปโหลดไฟล์วิดีโอเพื่อทดสอบวิเคราะห์อาการหลับในผ่านเว็บ")

# ดาวน์โหลด Model อัตโนมัติถ้ายังไม่มีในระบบ
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)


@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


detector = load_detector()


def euclidean_dist(pt1, pt2):
    return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)


def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    v1 = euclidean_dist(p2, p6)
    v2 = euclidean_dist(p3, p5)
    h = euclidean_dist(p1, p4)
    return (v1 + v2) / (2.0 * h)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

uploaded_file = st.file_uploader(
    "เลือกไฟล์วิดีโอ (.mp4, .mov)", type=["mp4", "mov"]
)

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()
    status_text = st.empty()

    CLOSED_COUNTER = 0

    st.info("🎬 กำลังประมวลผลวิดีโอ...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            avg_ear = (
                calculate_ear(landmarks, LEFT_EYE)
                + calculate_ear(landmarks, RIGHT_EYE)
            ) / 2.0

            if avg_ear < 0.21:
                CLOSED_COUNTER += 1
            else:
                CLOSED_COUNTER = 0

            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if CLOSED_COUNTER >= 10:
                cv2.putText(
                    frame,
                    "!!! DROWSINESS DETECTED !!!",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                status_text.error(
                    "🚨 ตรวจพบอาการหลับใน! (ผู้ขับปิดตาค้างไว้เกินกำหนด)"
                )
            else:
                status_text.success("✅ สภาพผู้ขับขี่ปกติ")

        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.balloons()
    st.success("🎉 วิเคราะห์วิดีโอเสร็จสิ้นเรียบร้อย!")

import math
import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)
from PIL import ImageFont, ImageDraw, Image
import av
import base64

st.set_page_config(page_title="ตรวจจับการหลับใน (เรียลไทม์)", layout="centered")
st.title("🚗 ระบบตรวจจับการหลับในขณะขับรถ (เรียลไทม์)")
st.caption("เปิดกล้องแล้วระบบจะส่งเสียงไซเรนเตือนทันทีที่ตรวจพบการหลับใน")

# ---------- 0. ฟอนต์ไทย ----------
FONT_PATH = "THSarabunNew.ttf"

def get_font():
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, 32)
    return ImageFont.load_default()

thai_font = get_font()

def put_thai_text(img_bgr, text, position, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=thai_font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- 1. โหลดโมเดล ----------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

@st.cache_resource
def get_model_path():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

model_path = get_model_path()

# ---------- 2. ค่าคงที่ EAR ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 10

def euclidean_dist(pt1, pt2):
    return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)

def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    v1 = euclidean_dist(p2, p6)
    v2 = euclidean_dist(p3, p5)
    h = euclidean_dist(p1, p4)
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

# ---------- 3. Video Processor ----------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.closed_counter = 0
        self.frame_idx = 0
        self.drowsy_flag = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.frame_idx += 1
        timestamp_ms = int(self.frame_idx * (1000 / 30))
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        status_text = "ไม่พบใบหน้า"
        status_color = (128, 128, 128)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                self.closed_counter += 1
            else:
                self.closed_counter = 0

            cv2.putText(img, f"EAR: {avg_ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if self.closed_counter >= CONSEC_FRAMES:
                status_text = "!!! ง่วงนอน โปรดหยุดพัก !!!"
                status_color = (0, 0, 255)
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 15)
                self.drowsy_flag = True
            else:
                status_text = "ปกติ"
                status_color = (0, 255, 0)
                self.drowsy_flag = False
        else:
            self.drowsy_flag = False

        img = put_thai_text(img, status_text, (20, h - 60), status_color)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- 4. WebRTC ----------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="drowsiness-realtime",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------- 5. HTML5 Audio Player (เล่นไฟล์เสียงตรงสไตล์ HTML) ----------
# ลิงก์ไฟล์เสียงไซเรนออนไลน์แบบ Direct
SOUND_URL = "https://www.soundjay.com/buttons/sounds/beep-07a.mp3"

audio_html = f"""
    <audio id="alarm-sound" src="{SOUND_URL}" preload="auto"></audio>
    <script>
        var audio = document.getElementById('alarm-sound');
        audio.play().catch(e => console.log('Autoplay blocked'));
    </script>
"""

# เช็คสถานะหลับตาจาก Frame
if ctx.video_processor:
    if ctx.video_processor.drowsy_flag:
        st.components.v1.html(audio_html, height=0)
        st.error("🚨 ตรวจพบการหลับใน! (กำลังเล่นเสียงเตือน)")

# ---------- 6. ปุ่มกดทดสอบเสียงเตือน ----------
if st.button("🔊 ทดสอบกดเล่นเสียงไซเรน"):
    st.components.v1.html(audio_html, height=0)
    st.success("ส่งคำสั่งเล่นเสียงเตือนเรียบร้อย!")

import math
import threading
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
    AudioProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    AudioHTMLAttributes,
    VideoHTMLAttributes,
)
from PIL import ImageFont, ImageDraw, Image
import av

st.set_page_config(page_title="ตรวจจับการหลับใน (เรียลไทม์)", layout="centered")
st.title("🚗 ระบบตรวจจับการหลับในขณะขับรถ (เรียลไทม์)")
st.caption("เปิดกล้อง+ไมค์ แล้วระบบจะส่งเสียงไซเรนอัตโนมัติทันทีที่หลับตานานเกินกำหนด")
st.info("⚠️ เบราว์เซอร์จะขอสิทธิ์กล้องและไมโครโฟน กรุณากด Allow ทั้งสองอย่าง (ระบบไม่บันทึกเสียงจากไมค์ไปที่ไหน ใช้แค่ส่งเสียงไซเรนกลับเท่านั้น)")

# ---------- 0. ฟอนต์ไทย ----------
FONT_PATH = "THSarabunNew.ttf"   # แก้ให้ตรงกับชื่อไฟล์ .ttf ที่อัปโหลดจริง
thai_font = ImageFont.truetype(FONT_PATH, 32)

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

# ---------- 3. สถานะแจ้งเตือนที่ใช้ร่วมกันระหว่างวิดีโอ/เสียง ----------
drowsy_event = threading.Event()   # set() = กำลังง่วง, clear() = ปกติ

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
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                drowsy_event.set()      # <-- สั่งให้แทร็กเสียงเริ่มดัง
            else:
                status_text = "ปกติ"
                status_color = (0, 255, 0)
                drowsy_event.clear()    # <-- สั่งให้เงียบ
        else:
            drowsy_event.clear()

        img = put_thai_text(img, status_text, (20, h - 60), status_color)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- 4. ตัวสร้างเสียงไซเรน ส่งผ่านแทร็กเสียงของ WebRTC โดยตรง ----------
class AlarmAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.phase_samples = 0

    def _make_tone_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        samples = frame.to_ndarray()
        n = samples.shape[-1]
        sr = frame.sample_rate

        if drowsy_event.is_set():
            freq = 1000.0
            t = (np.arange(n) + self.phase_samples) / sr
            tone = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            self.phase_samples += n
            out = np.tile(tone, (samples.shape[0], 1)) if samples.ndim == 2 else tone
        else:
            self.phase_samples = 0
            out = np.zeros_like(samples)

        new_frame = av.AudioFrame.from_ndarray(out, layout=frame.layout.name)
        new_frame.sample_rate = sr
        new_frame.pts = frame.pts
        return new_frame

    async def recv_queued(self, frames: list[av.AudioFrame]) -> list[av.AudioFrame]:
        return [self._make_tone_frame(f) for f in frames]
# ---------- 5. TURN server ----------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": st.secrets.get("TURN_USERNAME", ""),
            "credential": st.secrets.get("TURN_CREDENTIAL", ""),
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": st.secrets.get("TURN_USERNAME", ""),
            "credential": st.secrets.get("TURN_CREDENTIAL", ""),
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": st.secrets.get("TURN_USERNAME", ""),
            "credential": st.secrets.get("TURN_CREDENTIAL", ""),
        },
    ]
})

ctx = webrtc_streamer(
    key="drowsiness-realtime",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    audio_processor_factory=AlarmAudioProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": True},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=False),
    audio_html_attrs=AudioHTMLAttributes(autoPlay=True, controls=True, muted=False),
    async_processing=True,
)
if ctx.state.playing:
    st.success("✅ ระบบกำลังทำงาน — เสียงไซเรนจะดังอัตโนมัติทันทีที่ตรวจพบหลับตา ไม่ต้องกดอะไรเพิ่ม")

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

st.set_page_config(page_title="ตรวจจับการหลับใน (เรียลไทม์)", layout="centered")
st.title("🚗 ระบบตรวจจับการหลับในขณะขับรถ (เรียลไทม์)")
st.caption("เปิดกล้อง แล้วกดปุ่มปลดล็อกเสียงไซเรนด้านล่างก่อนใช้งาน")

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
                # วาดกรอบสีแดงหนาเพื่อส่งสัญญาณให้เบราว์เซอร์
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 30)
            else:
                status_text = "ปกติ"
                status_color = (0, 255, 0)

        img = put_thai_text(img, status_text, (20, h - 60), status_color)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- 4. WebRTC ----------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

webrtc_streamer(
    key="drowsiness-realtime",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------- 5. JavaScript ระบบเสียงไซเรนแบบเปิดสวิตช์ด้วยปุ่ม ----------
js_siren_engine = """
<div style="text-align: center; margin-top: 10px;">
    <button id="enable-audio-btn" style="
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">🔊 กดตรงนี้ 1 ครั้ง เพื่อเปิดระบบเสียงไซเรนเตือน</button>
    <p id="audio-status" style="color: gray; font-size: 14px; margin-top: 8px;">สถานะระบบเสียง: รอการเปิดใช้งาน</p>
</div>

<script>
let audioCtx = null;
let isAudioEnabled = false;
let lastPlayTime = 0;

const btn = document.getElementById('enable-audio-btn');
const statusText = document.getElementById('audio-status');

btn.addEventListener('click', () => {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    isAudioEnabled = true;
    btn.style.backgroundColor = '#28a745';
    btn.innerText = '✅ ระบบเสียงไซเรนพร้อมทำงานแล้ว!';
    statusText.innerText = 'สถานะระบบเสียง: กำลังเฝ้าระวังการหลับตา...';
    
    // ทดสอบยิงเสียงสั้นๆ เพื่อยืนยันว่าเสียงออกลำโพง
    playBeep();
});

function playBeep() {
    if (!audioCtx) return;
    let osc = audioCtx.createOscillator();
    let gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.15);
}

function triggerSiren() {
    if (!audioCtx || !isAudioEnabled) return;
    let now = Date.now();
    if (now - lastPlayTime < 300) return; // ป้องกันเสียงยิงซ้ำถี่เกินไป
    lastPlayTime = now;

    let osc = audioCtx.createOscillator();
    let gain = audioCtx.createGain();
    
    // เสียงไซเรนความถี่สลับสูง-ต่ำ
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(900, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(400, audioCtx.currentTime + 0.25);
    
    gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.start();
    osc.stop(audioCtx.currentTime + 0.25);
}

// ตรวจจับภาพขอบสีแดงบนหน้าจอวิดีโอ real-time
setInterval(() => {
    if (!isAudioEnabled) return;
    
    let videos = parent.document.querySelectorAll("video");
    if (videos.length === 0) return;
    let video = videos[0];
    if (video.paused || video.ended) return;

    let canvas = document.createElement("canvas");
    canvas.width = 40;
    canvas.height = 40;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 40, 40);
    
    // เช็คจุดสีแดงบริเวณขอบบนซ้ายของวิดีโอ
    let pixel = ctx.getImageData(3, 3, 1, 1).data;
    if (pixel[0] > 180 && pixel[1] < 50 && pixel[2] < 50) {
        triggerSiren();
    }
}, 80);
</script>
"""

st.components.v1.html(js_siren_engine, height=120)

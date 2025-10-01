# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Sign Language Alphabet Detection", layout="centered")

st.title("🖐 Sign Language Alphabet Detection (real-time)")
st.markdown("Camera will start automatically... show a hand sign!")

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.last_prediction = "No Hand"
        self._lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)
        prediction = "No Hand"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                try:
                    prediction = model.predict([coords])[0]
                except Exception:
                    prediction = "Err"

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(img, prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with self._lock:
            self.last_prediction = prediction

        return img


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    desired_playing_state=True,     # 👈 auto-start video
    show_controls=False             # 👈 hide Start/Stop buttons
)

if webrtc_ctx and webrtc_ctx.video_processor:
    st.markdown(f"**Prediction:** `{webrtc_ctx.video_processor.last_prediction}`")

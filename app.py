# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Video processor (replacement for while True loop)
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.prediction = "No Hand"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Mirror like cv2.flip
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)
        self.prediction = "No Hand"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract features (63 values)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                # Predict sign
                self.prediction = model.predict([coords])[0]

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Draw prediction text on frame
        cv2.putText(img, self.prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


# Streamlit app
st.title("🖐 Sign Language Alphabet Detection")
st.markdown("Show your hand sign in front of the webcam 👇 (real-time)")

# STUN server config for WebRTC (needed for cloud)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start WebRTC video stream
webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

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

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Define video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7,
                                    min_tracking_confidence=0.7)

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

                # Predict letter
                prediction = model.predict([coords])[0]

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Show prediction text
        cv2.putText(img, prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


# Streamlit UI
st.title("🖐 Sign Language Alphabet Detection")
st.markdown("Show your hand sign in front of the camera 👇")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="sign-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

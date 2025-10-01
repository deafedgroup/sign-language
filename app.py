import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Sign Language Alphabet Detection", layout="centered")

st.title("🖐 Sign Language Alphabet Detection (real-time)")
st.markdown("Allow camera access in your browser and show a hand sign. (This uses streamlit-webrtc for live video.)")

# Load trained model (ensure model.pkl is in the app root)
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("model.pkl not found. Upload or place your trained model named 'model.pkl' in the app root.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Mediapipe setup (module-level)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageProcessor(VideoProcessorBase):
    """
    VideoProcessor that is called for each incoming video frame.
    It behaves like your original while-loop: detect hand, extract 63 features,
    run model.predict([...]) and draw landmarks + predicted label on the frame.
    """
    def __init__(self):
        # Initialize Mediapipe Hands once
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.last_prediction = "No Hand"
        self._lock = threading.Lock()

    def recv(self, frame):
        """
        Called for each video frame. `frame` is an av.VideoFrame wrapped by streamlit-webrtc.
        Return a numpy ndarray (bgr24) with drawings and text.
        """
        # Convert to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Mirror to match webcam preview behavior like cv2.flip(frame, 1)
        img = cv2.flip(img, 1)

        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with Mediapipe
        result = self.hands.process(rgb)

        prediction = "No Hand"

        if result.multi_hand_landmarks:
            # If multiple hands, we'll take the last predicted label (like your original loop)
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract 63 features (x, y, z) for each of 21 landmarks
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                # Model prediction - safe call
                try:
                    pred = model.predict([coords])[0]
                    prediction = str(pred)
                except Exception as e:
                    # If model prediction fails, show error string on frame
                    prediction = "Err"
                    # Optionally: you can log to console for debugging
                    print("Prediction error:", e)

                # Draw Mediapipe landmarks & connections on the frame (in BGR)
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw prediction text on the frame (similar to your original cv2.putText)
        cv2.putText(img, prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Save last prediction (thread-safe)
        with self._lock:
            self.last_prediction = prediction

        # Return the modified frame (bgr24)
        return img

# WebRTC / STUN configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start streaming. This replaces your while True / cap.read loop.
webrtc_ctx = webrtc_streamer(
    key="sign-language-webrtc",
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,  # keep UI responsive; frames processed asynchronously
)

# Optionally show the last prediction below the video (reads from the processor instance)
placeholder = st.empty()
if webrtc_ctx.video_processor:
    # If the processor exists, display the last prediction live (updates on reruns)
    try:
        pred = webrtc_ctx.video_processor.last_prediction
        placeholder.markdown(f"**Last prediction:** `{pred}`")
    except Exception:
        placeholder.markdown("**Last prediction:** `-`")
else:
    placeholder.markdown("**Last prediction:** `-`")

st.markdown(
    """
**Notes**
- Make sure to allow camera permission in your browser.
- On Streamlit Cloud, include `packages.txt` with `libgl1` and `libglib2.0-0`, and add `streamlit-webrtc` to your `requirements.txt`.
"""
)

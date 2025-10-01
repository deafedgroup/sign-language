# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.title("🖐 Sign Language Alphabet Detection")
st.markdown("Camera starts automatically... show a hand sign!")

FRAME_WINDOW = st.image([])

# Open webcam
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("⚠️ Could not access webcam")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    prediction = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract 63 features (x,y,z for each of 21 landmarks)
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Predict letter
            prediction = model.predict([coords])[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display prediction on frame
    cv2.putText(frame, prediction, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

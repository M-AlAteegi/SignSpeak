"""
dataset.py
----------
Data collection script for building the ASL dataset.

Features:
- Captures hand landmarks from webcam using Mediapipe.
- Allows user to press keys (A–Z) to label current hand posture.
- Saves collected samples into 'asl_dataset.csv' in /other.

Usage:
Run this file to collect and expand your training dataset before running Modeltraining.py.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

data = []
labels = []

asl_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("Press the corresponding letter key to save the hand landmarks for that letter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            cv2.putText(frame, "Press Letter Key", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:  
                char_key = chr(key).upper()
                if char_key in asl_letters:
                    print(f"Saved letter: {char_key}")
                    data.append(landmarks)
                    labels.append(char_key)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('='):
        break

cap.release()
cv2.destroyAllWindows()

df_new = pd.DataFrame(data)
df_new['label'] = labels

df_new.to_csv("asl_dataset.csv", index=False)

print("Dataset saved successfully! (Overwritten)")
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model/gesture_model.h5')
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Adjust as per your dataset

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            # NOTE: Add logic to extract keypoints and predict gesture here

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

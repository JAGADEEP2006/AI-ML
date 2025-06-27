import cv2
import os
import mediapipe as mp

# Setup
label = input("Enter the label (e.g., A, B, C): ").upper()
save_dir = f'dataset/{label}'
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = 0

print(f"Collecting data for label: {label}. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Crop and save hand region
            h, w, _ = frame.shape
            hand = frame
            filename = os.path.join(save_dir, f"{label}_{count}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY))
            count += 1

    cv2.imshow("Collecting Hand Gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Saved {count} images to {save_dir}")
cap.release()
cv2.destroyAllWindows()

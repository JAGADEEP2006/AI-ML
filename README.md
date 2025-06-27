# Real-Time Sign Language Recognition

## Setup
```bash
pip install -r requirements.txt
```

## Run Hand Tracking Demo
```bash
python mediapipe_hand.py
```

## Train the Model
Put your labeled gesture images in `dataset/` folder (A-Z folders). Then run:
```bash
python train_model.py
```

## Run Gesture Recognition
```bash
python recognize_gesture.py
```

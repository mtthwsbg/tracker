# Hand + Face Tracker
Live webcam tracker (Python + MediaPipe). Hands: white boxes (hand + palm), red skeleton, finger count (fist = 0). Face: Happy / Neutral / Sad.

Run: double-click `run.cmd` (q or Esc quits; `--camera 1` picks another camera; `--selftest` checks camera + models).
Setup: venv at %LOCALAPPDATA%\vision-tracker-venv with `pip install -r requirements.txt`; download `hand_landmarker.task` and `face_landmarker.task` from storage.googleapis.com/mediapipe-models into `models/`.
Tuning: HAPPY_SMILE and SAD_SCORE at the top of tracker.py.

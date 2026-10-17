"""Live hand + face tracker.

Hands: white box around hand and palm, red skeleton, live finger count (fist = 0).
Face:  Happy / Neutral / Sad.
Keys:  q or Esc quits.   Usage: tracker.py [--camera N] [--selftest]
"""
import argparse
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(HERE, "models", "hand_landmarker.task")
FACE_MODEL = os.path.join(HERE, "models", "face_landmarker.task")

# --- tuning knobs (adjust to your face) ---
HAPPY_SMILE = 0.40      # avg mouthSmile blendshape above this -> Happy
SAD_SCORE = 0.30        # frown + inner-brow + mouth-press score above this -> Sad
SMOOTH_FRAMES = 5

WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 220, 0)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
PALM_POINTS = [0, 1, 5, 9, 13, 17]


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def count_fingers(pts):
    """pts: list of 21 (x, y) pixel points. Orientation-independent."""
    wrist = pts[0]
    n = 0
    # index, middle, ring, pinky: tip farther from wrist than the PIP joint
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if dist(pts[tip], wrist) > dist(pts[pip], wrist) * 1.05:
            n += 1
    # thumb: tip farther from pinky knuckle than the thumb IP joint
    if dist(pts[4], pts[17]) > dist(pts[3], pts[17]) * 1.05:
        n += 1
    return n


def bbox(pts, pad, w, h):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0),
            min(int(max(xs)) + pad, w - 1), min(int(max(ys)) + pad, h - 1))


def text(img, s, org, scale, color, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def classify_face(bs):
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    sad = (0.5 * (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
           + 0.3 * bs.get("browInnerUp", 0)
           + 0.2 * (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2)
    if smile > HAPPY_SMILE:
        return "Happy"
    if sad > SAD_SCORE:
        return "Sad"
    return "Neutral"


def majority(buf, default):
    return max(set(buf), key=list(buf).count) if buf else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="grab ~60 frames, print stats, exit")
    args = ap.parse_args()

    for m in (HAND_MODEL, FACE_MODEL):
        if not os.path.exists(m):
            raise SystemExit(f"Missing model: {m}")

    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    count_hist = {}                     # handedness label -> deque of counts
    face_hist = deque(maxlen=SMOOTH_FRAMES)
    t0 = time.time()
    frames = hand_frames = face_frames = 0
    fps = 0.0
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - t0) * 1000)

        hres = hands.detect_for_video(mp_img, ts)
        fres = face.detect_for_video(mp_img, ts)
        frames += 1

        # ---- hands ----
        total = 0
        seen = set()
        for i, lms in enumerate(hres.hand_landmarks):
            label = hres.handedness[i][0].category_name
            pts = [(l.x * w, l.y * h) for l in lms]
            seen.add(label)
            hist = count_hist.setdefault(label, deque(maxlen=SMOOTH_FRAMES))
            hist.append(count_fingers(pts))
            n = majority(hist, 0)
            total += n

            x1, y1, x2, y2 = bbox(pts, 20, w, h)                       # whole hand
            cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)
            px1, py1, px2, py2 = bbox([pts[j] for j in PALM_POINTS], 6, w, h)  # palm
            cv2.rectangle(frame, (px1, py1), (px2, py2), WHITE, 2)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), RED, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, RED, -1, cv2.LINE_AA)
            text(frame, str(n), (x1, max(y1 - 10, 50)), 1.8, WHITE, 3)
        for k in list(count_hist):
            if k not in seen:
                del count_hist[k]
        if hres.hand_landmarks:
            hand_frames += 1
        if len(hres.hand_landmarks) == 2:
            text(frame, f"Total: {total}", (10, 40), 1.0, WHITE)

        # ---- face ----
        if fres.face_landmarks:
            face_frames += 1
            bs = {c.category_name: c.score for c in fres.face_blendshapes[0]}
            face_hist.append(classify_face(bs))
            mood = majority(face_hist, "Neutral")
            fpts = [(l.x * w, l.y * h) for l in fres.face_landmarks[0]]
            fx1, fy1, fx2, fy2 = bbox(fpts, 10, w, h)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), GREEN, 1)
            text(frame, mood, (fx1, min(fy2 + 35, h - 10)), 1.1, GREEN, 2)
        else:
            face_hist.clear()

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        text(frame, f"{fps:.0f} fps", (w - 110, 30), 0.7, WHITE, 2)

        if args.selftest:
            if frames >= 60:
                break
            continue
        cv2.imshow("Hand + Face Tracker (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty("Hand + Face Tracker (q to quit)", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    if args.selftest:
        secs = time.time() - t0
        print(f"camera OK, {frames} frames in {secs:.1f}s ({frames / secs:.1f} fps), "
              f"frames with hand: {hand_frames}, with face: {face_frames}")


if __name__ == "__main__":
    main()

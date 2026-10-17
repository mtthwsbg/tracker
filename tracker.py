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
SAD_DELTA = 0.15        # sad score must beat your neutral baseline by this -> Sad (lower = more sensitive)
SMOOTH_FRAMES = 7
SIDEBAR_W = 280
CALIB_FRAMES = 30
PITCH_RANGE = 30.0      # degrees of up/down head tilt that reads as +/-100
YAW_RANGE = 40.0        # degrees of left/right head turn that reads as +/-100
TILT_SAD_COMP = 0.35    # how much looking up is discounted from the sad score (raise if looking up still reads Sad)
PALM_MIN = 0.25         # how square-on (palm or back) the hand must face the camera to be counted (0-1)

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


FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))  # thumb..pinky joints
PALM_LINES = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
UP_ON, UP_OFF = 0.80, 0.65     # finger straightness needed to switch on / to drop off (hysteresis stops flicker)


def d3(a, b):
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))


def finger_states(p3, prev):
    """p3: 21 (x, y, z) points. A finger is 'up' when its joints form a straight chain (tip-to-base / chain length).
    Straightness is 3-D, so tilting the hand doesn't fake a curl; curled fingers score low."""
    out = []
    for k, (m, a, b, t) in enumerate(FINGERS):
        chain = d3(p3[m], p3[a]) + d3(p3[a], p3[b]) + d3(p3[b], p3[t])
        r = d3(p3[m], p3[t]) / max(chain, 1e-6)
        up = r > (UP_OFF if prev[k] else UP_ON)
        if k == 0:      # thumb must also point away from the palm
            up = up and d3(p3[4], p3[17]) > d3(p3[3], p3[17]) * 1.05
        out.append(bool(up))
    return out


def bbox(pts, pad, w, h):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0),
            min(int(max(xs)) + pad, w - 1), min(int(max(ys)) + pad, h - 1))


def text(img, s, org, scale, color, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def face_scores(bs, fpts):
    """Returns (smile, sad) raw scores. Sad mixes blendshapes with mouth-corner droop."""
    avg = lambda a, b: (bs.get(a, 0) + bs.get(b, 0)) / 2
    smile = avg("mouthSmileLeft", "mouthSmileRight")
    # mouth corners (61, 291) below the lip centre (13, 14) = frown; scaled by face height
    face_h = max(fpts[152][1] - fpts[10][1], 1.0)
    lip_mid = (fpts[13][1] + fpts[14][1]) / 2
    droop = ((fpts[61][1] + fpts[291][1]) / 2 - lip_mid) / face_h
    sad = (1.0 * avg("mouthFrownLeft", "mouthFrownRight")
           + 0.5 * bs.get("browInnerUp", 0)
           + 0.3 * avg("browDownLeft", "browDownRight")
           + 0.5 * bs.get("mouthShrugLower", 0)
           + 0.3 * avg("mouthPressLeft", "mouthPressRight")
           + 8.0 * max(droop, 0))
    return smile, sad


def classify_face(smile, sad, sad_base):
    if smile > HAPPY_SMILE:
        return "Happy"
    if sad - sad_base > SAD_DELTA:
        return "Sad"
    return "Neutral"


def head_pose(lms, w, h):
    """(pitch_up_deg, yaw_right_deg) from 3-D face landmarks. Mirrored feed: +yaw = toward the right of the screen."""
    P = lambda i: (lms[i].x * w, lms[i].y * h, lms[i].z * w)
    top, chin = P(10), P(152)
    a, b = P(234), P(454)
    lc, rc = (a, b) if a[0] <= b[0] else (b, a)          # left / right cheek in the image
    pitch = np.degrees(np.arctan2(top[2] - chin[2], chin[1] - top[1]))   # chin nearer camera = looking up
    yaw = np.degrees(np.arctan2(rc[2] - lc[2], rc[0] - lc[0]))           # right cheek farther = turned right
    return float(pitch), float(yaw)


def hand_countable(pts, face_box):
    """Palm OR back of the hand toward the camera counts; edge-on hands and hands resting on the face don't."""
    v1 = (pts[5][0] - pts[0][0], pts[5][1] - pts[0][1])
    v2 = (pts[17][0] - pts[0][0], pts[17][1] - pts[0][1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    flat = abs(cross) / max(np.hypot(*v1) * np.hypot(*v2), 1e-6) > PALM_MIN   # ~0 when edge-on
    if face_box is not None:                       # hand's centre inside the face box = resting on the face
        cx = sum(q[0] for q in pts) / len(pts)
        cy = sum(q[1] for q in pts) / len(pts)
        if face_box[0] < cx < face_box[2] and face_box[1] < cy < face_box[3]:
            return False
    return flat


def draw_panel(h, fps, mood, smile, sad_rel, tilt, counts, calib):
    """Right-hand sidebar (SIDEBAR_W px wide). tilt = (up/down, left/right) in -100..100 or None."""
    p = np.full((h, SIDEBAR_W, 3), (30, 30, 30), np.uint8)
    cv2.line(p, (0, 0), (0, h), (90, 90, 90), 2)
    GREY, DIM = (190, 190, 190), (120, 120, 120)

    def header(s, y):
        text(p, s, (16, y), 0.6, (160, 160, 160), 1)
        cv2.line(p, (16, y + 8), (SIDEBAR_W - 16, y + 8), (70, 70, 70), 1)

    def bar(v, y):      # centred -100..100 bar
        x0, x1 = 16, SIDEBAR_W - 16
        mid = (x0 + x1) // 2
        cv2.rectangle(p, (x0, y), (x1, y + 6), (60, 60, 60), -1)
        cv2.line(p, (mid, y - 2), (mid, y + 8), (150, 150, 150), 1)
        if v is not None:
            xe = mid + int(v / 100 * (x1 - mid))
            cv2.rectangle(p, (min(mid, xe), y), (max(mid, xe), y + 6), (0, 200, 255), -1)

    text(p, f"{fps:.0f} FPS", (16, 32), 0.8, WHITE, 2)

    header("FACE", 66)
    colors = {"Happy": (0, 220, 0), "Sad": (255, 140, 40), "Neutral": WHITE}
    text(p, mood if mood else "No face", (16, 102), 1.1, colors.get(mood, DIM), 2)
    if mood:
        text(p, f"smile {smile:.2f}   sad {sad_rel:+.2f} / {SAD_DELTA:.2f}", (16, 124), 0.45, (170, 170, 170), 1)

    header("TILT  (-100..100)", 156)
    ud, lr = tilt if tilt else (None, None)
    text(p, "Up / Down", (16, 190), 0.6, GREY, 1)
    text(p, "--" if ud is None else f"{ud:+d}", (170, 190), 0.9, WHITE if tilt else DIM, 2)
    bar(ud, 200)
    text(p, "Left / Right", (16, 230), 0.6, GREY, 1)
    text(p, "--" if lr is None else f"{lr:+d}", (170, 230), 0.9, WHITE if tilt else DIM, 2)
    bar(lr, 240)

    header("HANDS", 280)
    y = 318
    for side in ("Left", "Right"):
        n = counts.get(side)
        text(p, f"{side}:", (16, y), 0.8, GREY, 1)
        text(p, "--" if n is None else str(n), (150, y), 1.2, WHITE if n is not None else DIM, 2)
        if side in counts and n is None:
            text(p, "paused", (190, y), 0.45, DIM, 1)
        y += 42
    shown = [v for v in counts.values() if v is not None]
    text(p, "Total:", (16, y + 4), 0.8, GREY, 1)
    text(p, str(sum(shown)) if shown else "--", (150, y + 4), 1.2, WHITE if shown else DIM, 2)

    text(p, "CALIBRATING - look straight" if calib else "c = calibrate (look straight)", (16, h - 34), 0.45, (0, 200, 255) if calib else (140, 140, 140), 1)
    text(p, "q / Esc = quit", (16, h - 12), 0.45, (140, 140, 140), 1)
    return p


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
    finger_state = {}                   # handedness label -> [thumb..pinky] up/down, for hysteresis
    face_hist = deque(maxlen=SMOOTH_FRAMES)
    sad_base = 0.0
    pitch0 = yaw0 = 0.0                 # 'looking straight' reference
    calib_buf = []                      # list of (sad, pitch, yaw) while calibrating, else None; starts calibrating
    tilt = None
    sad_rel = 0.0
    face_box = None                     # last seen face box, used to ignore a hand resting on the face
    mood = None
    smile = sad = 0.0
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
        counts = {}
        seen = set()
        for i, lms in enumerate(hres.hand_landmarks):
            # MediaPipe's label is swapped for this mirrored feed, so flip it
            label = "Right" if hres.handedness[i][0].category_name == "Left" else "Left"
            pts = [(l.x * w, l.y * h) for l in lms]
            p3 = [(l.x * w, l.y * h, l.z * w) for l in lms]
            seen.add(label)
            hist = count_hist.setdefault(label, deque(maxlen=SMOOTH_FRAMES))
            prev = finger_state.setdefault(label, [False] * 5)
            states = finger_states(p3, prev)
            finger_state[label] = states
            if hand_countable(pts, face_box):
                hist.append(sum(states))
                counts[label] = majority(hist, 0)
            else:
                hist.clear()                    # edge-on or resting on the face: don't count
                counts[label] = None

            # draw only what is really visible: palm + straight (raised) fingers
            shown = set(PALM_POINTS)
            lines = list(PALM_LINES)
            for up, (m, a, b, t) in zip(states, FINGERS):
                if up:
                    shown.update((m, a, b, t))
                    lines += [(m, a), (a, b), (b, t)]
            x1, y1, x2, y2 = bbox([pts[j] for j in shown], 20, w, h)    # whole hand (visible parts)
            cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)
            px1, py1, px2, py2 = bbox([pts[j] for j in PALM_POINTS], 6, w, h)  # palm
            cv2.rectangle(frame, (px1, py1), (px2, py2), WHITE, 2)
            for a, b in lines:
                cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), RED, 2, cv2.LINE_AA)
            for j in shown:
                cv2.circle(frame, (int(pts[j][0]), int(pts[j][1])), 4, RED, -1, cv2.LINE_AA)
        for k in list(count_hist):
            if k not in seen:
                del count_hist[k]
                finger_state.pop(k, None)
        if hres.hand_landmarks:
            hand_frames += 1
        # ---- face ----
        if fres.face_landmarks:
            face_frames += 1
            bs = {c.category_name: c.score for c in fres.face_blendshapes[0]}
            fpts = [(l.x * w, l.y * h) for l in fres.face_landmarks[0]]
            smile, sad = face_scores(bs, fpts)
            pitch, yaw = head_pose(fres.face_landmarks[0], w, h)
            if calib_buf is not None:
                calib_buf.append((sad, pitch, yaw))
                if len(calib_buf) >= CALIB_FRAMES:
                    sad_base, pitch0, yaw0 = (float(v) for v in np.mean(calib_buf, axis=0))
                    calib_buf = None
            pn = float(np.clip((pitch - pitch0) / PITCH_RANGE, -1, 1))
            yn = float(np.clip((yaw - yaw0) / YAW_RANGE, -1, 1))
            tilt = (int(round(pn * 100)), int(round(yn * 100)))
            sad_eff = sad - TILT_SAD_COMP * max(pn, 0)      # looking up bends the lips into a fake frown
            sad_rel = sad_eff - sad_base
            face_hist.append(classify_face(smile, sad_eff, sad_base))
            mood = majority(face_hist, "Neutral")
            fx1, fy1, fx2, fy2 = bbox(fpts, 10, w, h)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), GREEN, 1)
            face_box = (fx1, fy1, fx2, fy2)
        else:
            face_hist.clear()
            mood = None
            tilt = None
            face_box = None

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        panel = draw_panel(h, fps, mood, smile, sad_rel, tilt, counts, calib_buf is not None)
        frame = np.hstack([frame, panel])

        if args.selftest:
            if frames >= 60:
                break
            continue
        cv2.imshow("Hand + Face Tracker (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            calib_buf = []
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

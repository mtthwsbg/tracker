# Hand + Face Tracker

Live webcam tracker in Python (MediaPipe + OpenCV). Runs offline on a normal laptop CPU.

- **Hands** (up to 2): white box around the hand and the palm, red skeleton on the raised fingers, live finger count (fist = 0).
  Hidden or curled fingers are not drawn or counted. Counting pauses when the hand is edge-on or resting on your face.
- **Face:** Happy / Neutral / Sad, with the sad score corrected for head tilt (looking up no longer fakes a frown).
- **Head tilt:** Up/Down and Left/Right gauges from -100 to 100 (0 = looking straight; view is mirrored like a selfie).
- A sidebar on the right with two tabs. FACE: expression and tilt. HANDS: left/right/total finger counts and each finger up or down. The window is resizable.

## Run it

Needs Python 3.9+ and a webcam. The two model files (about 11 MB) download automatically on first run.

**Windows:** double-click `setup.cmd` once, then `run.cmd`.

**Mac / Linux:**
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tracker.py
```

Options: `--camera 1` (pick another camera), `--selftest` (grab 60 frames headless and print stats).

## Keys
- `c` - calibrate: look straight with a relaxed face (also happens automatically at start)
- `q` or `Esc` - quit
- `Tab`, `1`, `2`, or click a tab - switch between FACE and HANDS
- `t` - turn the pinch trace on or off (default on). The trace only draws while you pinch thumb and index together
- `o` - track an object: drag a box around it on the video. The box rotates and resizes with it (feature matching), HANDS tab
- `a` - auto-detect the object in your hand (common items only: cup, phone, bottle, scissors, remote, book...), HANDS tab

## Tuning (top of `tracker.py`)
| Constant | Effect |
|---|---|
| `HAPPY_SMILE` | smile score needed for Happy (lower = easier) |
| `SAD_DELTA` | how far above your neutral the sad score must go (lower = more sensitive) |
| `TILT_SAD_COMP` | how much looking up is discounted from "sad" |
| `PITCH_RANGE`, `YAW_RANGE` | degrees of head tilt that read as +/-100 |
| `UP_ON`, `UP_OFF` | finger straightness to switch on / off (lower = counts more easily) |
| `PALM_MIN` | how square-on the hand must face the camera to be counted |

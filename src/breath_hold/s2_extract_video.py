"""Re-verification helper: dump frames of MithinBreathHold-TimeStamps.mp4 so the transcribed laps (results/breath_hold_s2/mithin/laps_extracted.csv) can be re-read.
Needs OpenCV (`pip install --user opencv-python-headless`; the only place it is used).   python3 -m src.breath_hold.s2_extract_video OUT_DIR [every_n=80]"""
import sys
from pathlib import Path
import cv2

VIDEO = "data/recordings/drive_2026-10-19/MithinBreathHold-TimeStamps.mp4"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); every = int(sys.argv[2]) if len(sys.argv) > 2 else 80
c = cv2.VideoCapture(VIDEO); i = 0
while True:
    ok, f = c.read()
    if not ok: break
    if i % every == 0 or i == 0: cv2.imwrite(str(out / f"frame_{i:04d}.png"), f)
    i += 1
print(f"{i} frames read, fps {c.get(cv2.CAP_PROP_FPS):.2f}")

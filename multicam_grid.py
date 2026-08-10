import cv2
import numpy as np
from ultralytics import YOLO
import time

# ─── Config ───────────────────────────────────────────────
MODEL     = "yolov8n.pt"
CONF      = 0.4
CAM_INDEX = 0
GRID_W    = 1280
GRID_H    = 720
# ──────────────────────────────────────────────────────────

CAMERAS = [
    {"id": 1, "label": "GATE ENTRY",    "color": (0, 255, 120)},
    {"id": 2, "label": "PARKING LOT",   "color": (0, 200, 255)},
    {"id": 3, "label": "CORRIDOR A",    "color": (255, 180, 0)},
    {"id": 4, "label": "MAIN LOBBY",    "color": (255, 60, 120)},
]

print("[INFO] Loading YOLOv8-nano...")
model = YOLO(MODEL)
print("[INFO] Model ready. Starting feed...")

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit()

prev_time  = time.time()
frame_num  = 0
total_objs = 0

print("[INFO] Press 'q' to quit.")

while True:
    ret, raw = cap.read()
    if not ret:
        break

    frame_num += 1

    # ─── Run YOLO once per frame ──────────────────────────
    results    = model(raw, conf=CONF, verbose=False)
    annotated  = results[0].plot()
    total_objs = len(results[0].boxes)

    # ─── FPS ─────────────────────────────────────────────
    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # ─── Tile size ───────────────────────────────────────
    tw = GRID_W  // 2
    th = GRID_H  // 2

    tiles = []
    for cam in CAMERAS:
        tile = cv2.resize(annotated, (tw, th))

        # Dark overlay header bar
        overlay = tile.copy()
        cv2.rectangle(overlay, (0, 0), (tw, 38), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, tile, 0.3, 0, tile)

        # Colored left accent bar
        cv2.rectangle(tile, (0, 0), (4, 38), cam["color"], -1)

        # Camera label
        cv2.putText(tile, f"CAM {cam['id']} | {cam['label']}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, cam["color"], 2)

        # LIVE badge top-right
        cv2.rectangle(tile, (tw - 68, 8), (tw - 8, 30), (0, 0, 180), -1)
        cv2.putText(tile, "LIVE", (tw - 60, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Object count bottom-left
        cv2.rectangle(tile, (0, th - 32), (160, th), (10, 10, 10), -1)
        cv2.putText(tile, f"Objects: {total_objs}",
                    (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        # Border
        cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), cam["color"], 1)

        tiles.append(tile)

    # ─── Assemble 2x2 grid ───────────────────────────────
    row1 = np.hstack([tiles[0], tiles[1]])
    row2 = np.hstack([tiles[2], tiles[3]])
    grid = np.vstack([row1, row2])

    # ─── Top status bar ──────────────────────────────────
    status_bar = np.zeros((44, GRID_W, 3), dtype=np.uint8)
    status_bar[:] = (18, 18, 18)

    cv2.putText(status_bar, "MULTICAM SURVEILLANCE SYSTEM  |  PHASE 1 DEMO",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    cv2.putText(status_bar, f"FPS: {fps:.1f}",
                (GRID_W - 200, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 120), 2)

    cv2.putText(status_bar, f"FRAME: {frame_num}",
                (GRID_W - 330, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (160, 160, 160), 1)

    final = np.vstack([status_bar, grid])

    cv2.imshow("MultiCam YOLOv8 — Campus Surveillance Demo", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"[INFO] Stopped. Total frames: {frame_num}")
        break

cap.release()
cv2.destroyAllWindows()
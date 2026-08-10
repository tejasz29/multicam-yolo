import cv2
from ultralytics import YOLO
import time

# ─── Config ───────────────────────────────────────────────
MODEL = "yolov8n.pt"        # nano = fastest, works on your laptop
CONF  = 0.4                 # confidence threshold (0-1)
# ──────────────────────────────────────────────────────────

print("[INFO] Loading YOLOv8-nano model...")
model = YOLO(MODEL)
print("[INFO] Model loaded. Opening webcam...")

cap = cv2.VideoCapture(0)   # 0 = default webcam

if not cap.isOpened():
    print("[ERROR] Webcam not found. Check connection.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # ─── Run YOLO inference ───────────────────────────────
    results = model(frame, conf=CONF, verbose=False)
    annotated = results[0].plot()   # draws boxes + labels

    # ─── FPS counter ─────────────────────────────────────
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ─── Object count ────────────────────────────────────
    count = len(results[0].boxes)
    cv2.putText(annotated, f"Objects: {count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ─── Camera label ────────────────────────────────────
    cv2.putText(annotated, "CAM 1 - LIVE", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    cv2.imshow("YOLOv8 - Phase 1 Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
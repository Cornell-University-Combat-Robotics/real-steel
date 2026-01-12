import cv2
import numpy as np
from ultralytics import YOLO

# Pose model
model = YOLO("yolo11x-pose.pt")

# Choose input: 0 for webcam, or "video.mp4"
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Could not open video source"

CONF_THRES = 0.25
KP_CONF_THRES = 0.5

# We'll track the hip-center across frames
prev_pos = None  # (x, y)

def hip_center(kpts_xy, kpts_conf):
    """
    kpts_xy: (17,2), kpts_conf: (17,)
    COCO: left_hip=11, right_hip=12
    Returns (x,y) or None if low confidence.
    """
    lh_i, rh_i = 11, 12
    if kpts_conf[lh_i] < KP_CONF_THRES or kpts_conf[rh_i] < KP_CONF_THRES:
        return None
    lh = kpts_xy[lh_i]
    rh = kpts_xy[rh_i]
    return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)

def bbox_area(box):
    # box: (x1,y1,x2,y2)
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: resize for speed (uncomment if needed)
    # frame = cv2.resize(frame, (960, 540))

    results = model.predict(
        frame,
        conf=CONF_THRES,
        iou=0.45,
        classes=[0],   # person
        verbose=False,
        device="cpu"     # GPU 0; use "cpu" if needed
    )[0]

    annotated = frame.copy()

    chosen_pos = None
    chosen_box = None

    if results.keypoints is not None and results.boxes is not None:
        kpts_xy_all = results.keypoints.xy.cpu().numpy()     # (N,17,2)
        kpts_conf_all = results.keypoints.conf.cpu().numpy() # (N,17)
        boxes = results.boxes.xyxy.cpu().numpy()             # (N,4)

        candidates = []
        for i in range(len(boxes)):
            pos = hip_center(kpts_xy_all[i], kpts_conf_all[i])
            if pos is None:
                continue
            area = bbox_area(boxes[i])
            candidates.append((i, pos, area))

        if candidates:
            # If we have a previous position, prefer the candidate closest to it.
            # Otherwise, pick the biggest (most likely the main subject).
            if prev_pos is not None:
                px, py = prev_pos
                candidates.sort(key=lambda t: ( (t[1][0]-px)**2 + (t[1][1]-py)**2 ))
            else:
                candidates.sort(key=lambda t: -t[2])

            best_i, best_pos, _ = candidates[0]
            chosen_pos = best_pos
            chosen_box = boxes[best_i]

    # Draw + update state
    if chosen_pos is not None:
        prev_pos = chosen_pos
        cx, cy = int(chosen_pos[0]), int(chosen_pos[1])

        # draw position
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(annotated, f"HipCenter ({cx},{cy})",
                    (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # draw chosen bbox
        x1, y1, x2, y2 = map(int, chosen_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    else:
        # lost target (no confident hips found)
        cv2.putText(annotated, "Target not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        prev_pos = None

    cv2.imshow("Single Person Pose Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint names (YOLO pose model uses COCO format)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def select_joint():
    """Prompt user to select a joint to track."""
    print("\n=== Joint Selection ===")
    print("Available joints:")
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {i}: {name}")
    print("\nSpecial options:")
    print("  17: hip_center (average of left and right hips)")
    print("  18: shoulder_center (average of left and right shoulders)")
    
    while True:
        try:
            choice = int(input("\nEnter joint number to track: "))
            if 0 <= choice <= 18:
                if choice == 17:
                    return "hip_center", None
                elif choice == 18:
                    return "shoulder_center", None
                else:
                    return KEYPOINT_NAMES[choice], choice
            else:
                print("Invalid choice. Please enter a number between 0 and 18.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_joint_position(kpts_xy, kpts_conf, joint_name, joint_idx):
    """
    Get position of the selected joint.
    Returns (x, y) or None if confidence is too low.
    """
    if joint_name == "hip_center":
        lh_i, rh_i = 11, 12
        if kpts_conf[lh_i] < KP_CONF_THRES or kpts_conf[rh_i] < KP_CONF_THRES:
            return None
        lh = kpts_xy[lh_i]
        rh = kpts_xy[rh_i]
        return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    
    elif joint_name == "shoulder_center":
        ls_i, rs_i = 5, 6
        if kpts_conf[ls_i] < KP_CONF_THRES or kpts_conf[rs_i] < KP_CONF_THRES:
            return None
        ls = kpts_xy[ls_i]
        rs = kpts_xy[rs_i]
        return ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    
    else:
        # Single keypoint
        if kpts_conf[joint_idx] < KP_CONF_THRES:
            return None
        return tuple(kpts_xy[joint_idx])

def bbox_area(box):
    """Calculate bounding box area."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

# Select joint at startup
joint_name, joint_idx = select_joint()
print(f"\nTracking: {joint_name}")
print("Press 'q' to quit\n")

# Pose model
model = YOLO("yolo11n-pose.pt")

# Choose input: 0 for webcam, or "video.mp4"
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Could not open video source"

CONF_THRES = 0.25
KP_CONF_THRES = 0.5

# Track the selected joint across frames
prev_pos = None  # (x, y)

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
        device="cpu"   # use "cpu" or GPU number
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
            pos = get_joint_position(kpts_xy_all[i], kpts_conf_all[i], joint_name, joint_idx)
            if pos is None:
                continue
            area = bbox_area(boxes[i])
            candidates.append((i, pos, area))

        if candidates:
            # If we have a previous position, prefer the candidate closest to it.
            # Otherwise, pick the biggest (most likely the main subject).
            if prev_pos is not None:
                px, py = prev_pos
                candidates.sort(key=lambda t: ((t[1][0]-px)**2 + (t[1][1]-py)**2))
            else:
                candidates.sort(key=lambda t: -t[2])

            best_i, best_pos, _ = candidates[0]
            chosen_pos = best_pos
            chosen_box = boxes[best_i]

    # Draw + update state
    if chosen_pos is not None:
        prev_pos = chosen_pos
        cx, cy = int(chosen_pos[0]), int(chosen_pos[1])

        # Draw position
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(annotated, f"{joint_name} ({cx},{cy})",
                    (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw chosen bbox
        x1, y1, x2, y2 = map(int, chosen_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    else:
        # Lost target
        cv2.putText(annotated, f"{joint_name} not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        prev_pos = None

    cv2.imshow("Single Person Pose Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
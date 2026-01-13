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

CONF_THRES = 0.25
KP_CONF_THRES = 0.5

def select_joints():
    """Prompt user to select one or more joints to track (comma-separated)."""
    print("\n=== Joint Selection ===")
    print("Available joints:")
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {i}: {name}")
    print("\nSpecial options:")
    print("  17: hip_center (average of left and right hips)")
    print("  18: shoulder_center (average of left and right shoulders)")
    print("\nExamples:")
    print("  0,1,2,3,4   -> nose + eyes + ears")
    print("  5,6,7,8,9,10 -> shoulders + elbows + wrists")
    print("  17,18       -> hip_center + shoulder_center")

    while True:
        raw = input("\nEnter joint numbers to track (comma-separated): ").strip()
        try:
            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            choices = [int(p) for p in parts]
        except ValueError:
            print("Invalid input. Use comma-separated integers like: 0,1,2,3,4")
            continue

        if not choices:
            print("Please enter at least one number.")
            continue

        if any(c < 0 or c > 18 for c in choices):
            print("Invalid choice(s). Use numbers between 0 and 18.")
            continue

        # Build targets list: (name, idx) where idx=None for special centers
        targets = []
        for c in choices:
            if c == 17:
                targets.append(("hip_center", None))
            elif c == 18:
                targets.append(("shoulder_center", None))
            else:
                targets.append((KEYPOINT_NAMES[c], c))

        # Remove duplicates while preserving order
        seen = set()
        uniq = []
        for t in targets:
            if t not in seen:
                uniq.append(t)
                seen.add(t)

        return uniq

def get_joint_position(kpts_xy, kpts_conf, joint_name, joint_idx):
    """
    Get position of a requested joint.
    Returns (x, y) or None if confidence is too low.
    """
    if joint_name == "hip_center":
        lh_i, rh_i = 11, 12
        if kpts_conf[lh_i] < KP_CONF_THRES or kpts_conf[rh_i] < KP_CONF_THRES:
            return None
        lh = kpts_xy[lh_i]
        rh = kpts_xy[rh_i]
        return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)

    if joint_name == "shoulder_center":
        ls_i, rs_i = 5, 6
        if kpts_conf[ls_i] < KP_CONF_THRES or kpts_conf[rs_i] < KP_CONF_THRES:
            return None
        ls = kpts_xy[ls_i]
        rs = kpts_xy[rs_i]
        return ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)

    # Single keypoint
    if kpts_conf[joint_idx] < KP_CONF_THRES:
        return None
    x, y = kpts_xy[joint_idx]
    return (float(x), float(y))

def bbox_area(box):
    """Calculate bounding box area."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def open_camera_macos():
    """
    macOS: AVFoundation backend is usually the most reliable.
    Try index 1 then 0.
    """
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    return cap

# ------------------ main ------------------

targets = select_joints()
print("\nTracking:", ", ".join([t[0] for t in targets]))
print("IMPORTANT: Press 'q' while the OpenCV window is focused to quit.\n")

model = YOLO("yolo11n-pose.pt")  # faster on CPU

cap = open_camera_macos()
assert cap.isOpened(), (
    "Could not open camera. On macOS, check:\n"
    "System Settings -> Privacy & Security -> Camera -> allow Terminal/VS Code.\n"
    "Also try changing the camera index."
)

prev_center = None  # (x,y) for keeping the same person

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        # Camera opened but no frames coming through
        print("ERROR: Camera opened but no frames received.")
        break

    # Optional speed boost (uncomment)
    # frame = cv2.resize(frame, (640, 360))

    r = model.predict(
        frame,
        conf=CONF_THRES,
        iou=0.45,
        classes=[0],      # person only
        verbose=False,
        device="cpu"
    )[0]

    annotated = frame.copy()

    chosen_box = None
    chosen_positions = None
    chosen_center = None

    if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
        kpts_xy_all = r.keypoints.xy.cpu().numpy()       # (N,17,2)
        kpts_conf_all = r.keypoints.conf.cpu().numpy()   # (N,17)
        boxes = r.boxes.xyxy.cpu().numpy()               # (N,4)

        candidates = []
        for i in range(len(boxes)):
            positions = {}
            for name, idx in targets:
                pos = get_joint_position(kpts_xy_all[i], kpts_conf_all[i], name, idx)
                if pos is not None:
                    positions[name] = pos

            # require at least one selected joint visible
            if not positions:
                continue

            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
            area = bbox_area(boxes[i])

            candidates.append((i, center, area, positions))

        if candidates:
            # Keep same person: nearest to previous center, else biggest
            if prev_center is not None:
                px, py = prev_center
                candidates.sort(key=lambda t: (t[1][0] - px) ** 2 + (t[1][1] - py) ** 2)
            else:
                candidates.sort(key=lambda t: -t[2])

            best_i, best_center, _, best_positions = candidates[0]
            chosen_box = boxes[best_i]
            chosen_positions = best_positions
            chosen_center = best_center

    if chosen_box is not None:
        prev_center = chosen_center

        # Draw bbox
        x1, y1, x2, y2 = map(int, chosen_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw each selected joint
        for name, (x, y) in chosen_positions.items():
            cx, cy = int(x), int(y)
            cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
            cv2.putText(
                annotated,
                f"{name} ({cx},{cy})",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    else:
        cv2.putText(
            annotated,
            "Selected joints not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        prev_center = None

    cv2.imshow("Multi-Joint Tracking (Single Person)", annotated)

    # NOTE: 'q' works ONLY when this OpenCV window is focused.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

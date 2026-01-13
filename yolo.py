import cv2
import numpy as np
from ultralytics import YOLO
import math

# COCO indices
KPT = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

CONF_THRES = 0.25
KP_CONF_THRES = 0.5

# Punch-distance normalization (tune if needed)
# d = (shoulder->wrist distance) / shoulder_width
D_MIN = 1.2   # relaxed arm-ish
D_MAX = 3.0   # fully extended-ish

# Smoothing (0..1). Higher = snappier.
SMOOTH_ALPHA = 0.25

def open_camera_macos():
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    return cap

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def lerp(a, b, t):
    return a + (b - a) * t

def get_kpt(kpts_xy, kpts_conf, name):
    idx = KPT[name]
    if kpts_conf[idx] < KP_CONF_THRES:
        return None
    x, y = kpts_xy[idx]
    return np.array([float(x), float(y)], dtype=np.float32)

def dist(a, b):
    return float(np.linalg.norm(a - b))

def bbox_area(box):
    return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))

def draw_bar(img, x, y, w, h, value_0_1, label):
    v = clamp(value_0_1, 0.0, 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    fill_w = int(w * v)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), (255, 255, 255), -1)
    cv2.putText(img, f"{label}: {v:.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def draw_centered_bar(img, x, y, w, h, value_minus1_1, label):
    v = clamp(value_minus1_1, -1.0, 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 2)
    mid = x + w // 2
    cv2.line(img, (mid, y), (mid, y + h), (0,0,0), 2)

    half = w // 2
    if v >= 0:
        fill = int(half * v)
        cv2.rectangle(img, (mid, y), (mid + fill, y + h), (255,255,255), -1)
    else:
        fill = int(half * (-v))
        cv2.rectangle(img, (mid - fill, y), (mid, y + h), (255,255,255), -1)

    cv2.putText(img, f"{label}: {v:.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def extension01(shoulder, wrist, shoulder_w):
    """
    Returns arm extension in [0,1] based on normalized shoulder->wrist distance.
    """
    if shoulder is None or wrist is None or shoulder_w is None or shoulder_w <= 1e-6:
        return None
    d = dist(shoulder, wrist) / shoulder_w
    return clamp((d - D_MIN) / (D_MAX - D_MIN), 0.0, 1.0)

# ---------------- Main ----------------

model = YOLO("yolo11n-pose.pt")  # fast on CPU

cap = open_camera_macos()
assert cap.isOpened(), (
    "Could not open camera. On macOS:\n"
    "System Settings -> Privacy & Security -> Camera -> allow Terminal/VS Code."
)

prev_center = None
smooth = {
    "throttle": 0.0,       # -1..1  (right arm forward = +, left arm forward = -)
    "steer": 0.0,          # -1..1  (shoulder tilt)
    "right_ext": 0.0,      # 0..1
    "left_ext": 0.0,       # 0..1
}

print("Controls:")
print("  Right arm extension -> throttle UP (+)")
print("  Left arm extension  -> throttle BACK (-)")
print("  Shoulder tilt        -> steering (-1..1)")
print("Focus the OpenCV window and press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: no frames received.")
        break

    # Optional speed-up:
    # frame = cv2.resize(frame, (640, 360))

    r = model.predict(
        frame,
        conf=CONF_THRES,
        iou=0.45,
        classes=[0],
        verbose=False,
        device="cpu"
    )[0]

    annotated = frame.copy()

    chosen_i = None
    chosen_box = None

    if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
        kpts_xy_all = r.keypoints.xy.cpu().numpy()
        kpts_conf_all = r.keypoints.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        candidates = []
        for i in range(len(boxes)):
            ls = get_kpt(kpts_xy_all[i], kpts_conf_all[i], "left_shoulder")
            rs = get_kpt(kpts_xy_all[i], kpts_conf_all[i], "right_shoulder")
            if ls is None or rs is None:
                continue
            center = (ls + rs) * 0.5
            area = bbox_area(boxes[i])
            candidates.append((i, center, area))

        if candidates:
            if prev_center is not None:
                px, py = prev_center
                candidates.sort(key=lambda t: (t[1][0]-px)**2 + (t[1][1]-py)**2)
            else:
                candidates.sort(key=lambda t: -t[2])

            chosen_i, center, _ = candidates[0]
            prev_center = (float(center[0]), float(center[1]))
            chosen_box = boxes[chosen_i]

    # Raw (unsmoothed) signals
    right_ext = None
    left_ext = None
    steer = 0.0
    throttle = 0.0

    if chosen_i is not None:
        kxy = r.keypoints.xy.cpu().numpy()[chosen_i]
        kcf = r.keypoints.conf.cpu().numpy()[chosen_i]

        ls = get_kpt(kxy, kcf, "left_shoulder")
        rs = get_kpt(kxy, kcf, "right_shoulder")
        lw = get_kpt(kxy, kcf, "left_wrist")
        rw = get_kpt(kxy, kcf, "right_wrist")

        shoulder_w = dist(ls, rs) if (ls is not None and rs is not None) else None

        # Arm extensions (0..1)
        left_ext = extension01(ls, lw, shoulder_w)
        right_ext = extension01(rs, rw, shoulder_w)

        # Throttle mapping you requested:
        #   right arm forward => + throttle
        #   left arm forward  => - throttle
        # If one arm missing, still works with the other.
        re = right_ext if right_ext is not None else 0.0
        le = left_ext if left_ext is not None else 0.0
        throttle = clamp(re - le, -1.0, 1.0)

        # Steering from shoulder-line tilt
        # v = rs - ls ; image y increases downward -> sign may feel inverted; flip if needed.
        v = rs - ls
        angle = math.degrees(math.atan2(v[1], v[0]))  # 0 = level shoulders
        steer = clamp(angle / 25.0, -1.0, 1.0)  # 25 degrees = full steer

        # Draw box and keypoints
        x1, y1, x2, y2 = map(int, chosen_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255,255,255), 2)

        for pt in [ls, rs, lw, rw]:
            if pt is not None:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 6, (255,255,255), -1)

    else:
        prev_center = None

    # Smooth
    smooth["right_ext"] = lerp(smooth["right_ext"], (right_ext if right_ext is not None else 0.0), SMOOTH_ALPHA)
    smooth["left_ext"]  = lerp(smooth["left_ext"],  (left_ext  if left_ext  is not None else 0.0), SMOOTH_ALPHA)
    smooth["throttle"]  = lerp(smooth["throttle"], throttle, SMOOTH_ALPHA)
    smooth["steer"]     = lerp(smooth["steer"], steer, SMOOTH_ALPHA)

    # HUD
    pad = 20
    bar_w = 320
    bar_h = 22
    y0 = pad + 40

    # Throttle centered bar (-1..1)
    draw_centered_bar(annotated, pad, y0, bar_w, bar_h, smooth["throttle"], "THROTTLE (R-L)")
    draw_centered_bar(annotated, pad, y0 + 50, bar_w, bar_h, smooth["steer"], "STEER (tilt)")
    draw_bar(annotated, pad, y0 + 100, bar_w, bar_h, smooth["right_ext"], "RIGHT ARM EXT")
    draw_bar(annotated, pad, y0 + 150, bar_w, bar_h, smooth["left_ext"], "LEFT ARM EXT")

    # Text summary
    cv2.putText(
        annotated,
        f"Throttle:{smooth['throttle']:+.2f}  Steer:{smooth['steer']:+.2f}  R:{smooth['right_ext']:.2f}  L:{smooth['left_ext']:.2f}",
        (pad, pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Real Steel Test (R arm=forward, L arm=back)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO

# COCO indices
KPT = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

# ---------------- TUNING ----------------
CONF_THRES = 0.25
KP_CONF_THRES = 0.5

# Steering from torso lean (shoulder_center vs hip_center), normalized by shoulder width
STEER_GAIN = 1.6          # lower = less sensitive
STEER_DEADZONE = 0.08

# Throttle: (right wrist extension) - (left wrist extension)
THROTTLE_GAIN = 1.8
THROTTLE_DEADZONE = 0.10

# Wrist extension measured from torso_center, normalized by shoulder width.
# Tune these two if throttle is always non-zero or never reaches max.
EXT_MIN = 0.55
EXT_MAX = 1.85

SMOOTH_ALPHA = 0.18
# ----------------------------------------

def open_camera_macos():
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    return cap

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def lerp(a, b, t):
    return a + (b - a) * t

def deadzone(x, dz):
    if abs(x) < dz:
        return 0.0
    return np.sign(x) * (abs(x) - dz) / (1.0 - dz)

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

def norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)

def draw_bar(img, x, y, w, h, value_0_1, label):
    v = clamp(value_0_1, 0.0, 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 2)
    fill_w = int(w * v)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), (0, 255, 0), -1)
    cv2.putText(img, f"{label}: {v:.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def draw_centered_bar(img, x, y, w, h, value_minus1_1, label):
    v = clamp(value_minus1_1, -1.0, 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 2)
    mid = x + w // 2
    cv2.line(img, (mid, y), (mid, y + h), (255,255,255), 2)

    half = w // 2
    if v >= 0:
        fill = int(half * v)
        cv2.rectangle(img, (mid, y), (mid + fill, y + h), (0, 255, 0), -1)
    else:
        fill = int(half * (-v))
        cv2.rectangle(img, (mid - fill, y), (mid, y + h), (0, 0, 255), -1)

    cv2.putText(img, f"{label}: {v:+.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ---------------- Main ----------------

model = YOLO("yolo11n-pose.pt")
cap = open_camera_macos()
assert cap.isOpened(), (
    "Could not open camera. On macOS:\n"
    "System Settings -> Privacy & Security -> Camera -> allow Terminal/VS Code."
)

prev_center = None
smooth = {"throttle": 0.0, "steer": 0.0, "right_ext": 0.0, "left_ext": 0.0}

print("\nThrottle motion:")
print("  + Forward: extend RIGHT arm away from torso (left stays closer).")
print("  - Backward: extend LEFT arm away from torso (right stays closer).")
print("Steer motion:")
print("  Lean your torso LEFT/RIGHT (shoulders shift relative to hips).")
print("Press 'q' in the OpenCV window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: no frames received.")
        break

    # IMPORTANT: mirror to fix left/right swap on selfie cameras
    frame = cv2.flip(frame, 1)

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
            lh = get_kpt(kpts_xy_all[i], kpts_conf_all[i], "left_hip")
            rh = get_kpt(kpts_xy_all[i], kpts_conf_all[i], "right_hip")
            if ls is None or rs is None or lh is None or rh is None:
                continue
            shoulder_c = (ls + rs) * 0.5
            hip_c = (lh + rh) * 0.5
            center = (shoulder_c + hip_c) * 0.5
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
    else:
        prev_center = None

    # Raw signals
    steer = 0.0
    throttle = 0.0
    right_ext01 = 0.0
    left_ext01 = 0.0

    if chosen_i is not None:
        kxy = r.keypoints.xy.cpu().numpy()[chosen_i]
        kcf = r.keypoints.conf.cpu().numpy()[chosen_i]

        ls = get_kpt(kxy, kcf, "left_shoulder")
        rs = get_kpt(kxy, kcf, "right_shoulder")
        lw = get_kpt(kxy, kcf, "left_wrist")
        rw = get_kpt(kxy, kcf, "right_wrist")
        lh = get_kpt(kxy, kcf, "left_hip")
        rh = get_kpt(kxy, kcf, "right_hip")

        x1, y1, x2, y2 = map(int, chosen_box)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,255,255), 2)

        shoulder_c = (ls + rs) * 0.5
        hip_c = (lh + rh) * 0.5
        torso_c = (shoulder_c + hip_c) * 0.5

        shoulder_w = dist(ls, rs)
        if shoulder_w < 1e-6:
            shoulder_w = None

        # STEER: shoulders shifting relative to hips (torso lean)
        if shoulder_w is not None:
            lean_x = float((shoulder_c[0] - hip_c[0]) / shoulder_w)
            steer = clamp(lean_x * STEER_GAIN, -1.0, 1.0)
            steer = deadzone(steer, STEER_DEADZONE)

        # THROTTLE: wrist extension relative to torso center (more stable)
        if shoulder_w is not None:
            if rw is not None:
                right_norm = dist(torso_c, rw) / shoulder_w
                right_ext01 = norm01(right_norm, EXT_MIN, EXT_MAX)
            if lw is not None:
                left_norm = dist(torso_c, lw) / shoulder_w
                left_ext01 = norm01(left_norm, EXT_MIN, EXT_MAX)

            throttle = clamp((right_ext01 - left_ext01) * THROTTLE_GAIN, -1.0, 1.0)
            throttle = deadzone(throttle, THROTTLE_DEADZONE)

        # Visualize keypoints + centers
        for pt in [ls, rs, lw, rw, lh, rh]:
            if pt is not None:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 6, (255,255,255), -1)

        cv2.circle(annotated, (int(shoulder_c[0]), int(shoulder_c[1])), 7, (0,255,255), -1)  # yellow
        cv2.circle(annotated, (int(hip_c[0]), int(hip_c[1])), 7, (255,0,255), -1)            # magenta
        cv2.circle(annotated, (int(torso_c[0]), int(torso_c[1])), 7, (0,200,200), -1)         # cyan-ish

        cv2.line(annotated, (int(hip_c[0]), int(hip_c[1])), (int(shoulder_c[0]), int(shoulder_c[1])),
                 (200,200,200), 2)

    # Smooth
    smooth["steer"] = lerp(smooth["steer"], steer, SMOOTH_ALPHA)
    smooth["throttle"] = lerp(smooth["throttle"], throttle, SMOOTH_ALPHA)
    smooth["right_ext"] = lerp(smooth["right_ext"], right_ext01, SMOOTH_ALPHA)
    smooth["left_ext"] = lerp(smooth["left_ext"], left_ext01, SMOOTH_ALPHA)

    # HUD
    pad = 20
    bar_w = 380
    bar_h = 22
    y0 = pad + 40

    draw_centered_bar(annotated, pad, y0, bar_w, bar_h, smooth["throttle"], "THROTTLE (R-L)")
    draw_centered_bar(annotated, pad, y0 + 50, bar_w, bar_h, smooth["steer"], "STEER (lean)")
    draw_bar(annotated, pad, y0 + 100, bar_w, bar_h, smooth["right_ext"], "RIGHT EXT (0..1)")
    draw_bar(annotated, pad, y0 + 150, bar_w, bar_h, smooth["left_ext"], "LEFT  EXT (0..1)")

    cv2.putText(
        annotated,
        f"T:{smooth['throttle']:+.2f}  S:{smooth['steer']:+.2f}  R:{smooth['right_ext']:.2f}  L:{smooth['left_ext']:.2f}",
        (pad, pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255,255,255),
        2
    )

    cv2.imshow("Real Steel Prototype (mirrored, hips+shoulders)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

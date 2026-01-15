import cv2
import numpy as np
import time
from ultralytics import YOLO
from main_helpers import get_motor_groups

# ================= BOT SETUP =================
ser1, motor_group_1, weapon_motor_group_1 = get_motor_groups(False, 1, 3, 4)

ENABLE_BOT2 = True
if ENABLE_BOT2:
    ser2, motor_group_2, weapon_motor_group_2 = get_motor_groups(False, 2, 5, 6)
else:
    ser2, motor_group_2, weapon_motor_group_2 = None, None, None

# ================= POSE SETTINGS =================
KPT = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

CONF_THRES = 0.18
KP_CONF_THRES = 0.35

STEER_GAIN = 1.4
STEER_DEADZONE = 0.10

THROTTLE_GAIN = 1.8
THROTTLE_DEADZONE = 0.10

EXT_MIN = 0.55
EXT_MAX = 1.85

THROTTLE_SCALE = 0.55
STEER_SCALE = 0.30

# ================= VIEW / SIGN FIXES =================
MIRROR_VIEW = True          # display-only
ARM_SWAP_MODE = False

THROTTLE_INVERT = False
MOTOR_THROTTLE_INVERT = False
STEER_INVERT = True

# ================= PERFORMANCE (M3-safe) =================
INFER_W, INFER_H = 512, 288
INFER_EVERY_N = 2
MOTOR_HZ = 25.0

DEVICE = "mps"        # on M3 this should be good for predict()
USE_HALF = False      # keep False on MPS

SMOOTH_TAU = 0.10
POSE_TIMEOUT_S = 0.35

DROP_OLD_FRAMES = True
GRAB_COUNT = 2        # how many grabs to drop per loop

# Player-stability without track()
MATCH_MAX_DIST = 0.22  # normalized by screen width (0.15–0.30 typical)
# ================= END TUNING =================

def open_camera_macos():
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def deadzone(x, dz):
    if abs(x) < dz:
        return 0.0
    return float(np.sign(x) * (abs(x) - dz) / (1.0 - dz))

def lerp(a, b, t):
    return a + (b - a) * t

def smooth_step(prev, target, dt, tau):
    if tau <= 1e-6:
        return target
    alpha = 1.0 - np.exp(-dt / tau)
    return lerp(prev, target, float(alpha))

def get_kpt(kxy, kcf, name):
    idx = KPT[name]
    if kcf[idx] < KP_CONF_THRES:
        return None
    x, y = kxy[idx]
    return np.array([float(x), float(y)], dtype=np.float32)

def dist(a, b):
    return float(np.linalg.norm(a - b))

def bbox_area(box):
    return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))

def norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)

def draw_centered_bar(img, x, y, w, h, value_minus1_1, label):
    v = clamp(value_minus1_1, -1.0, 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    mid = x + w // 2
    cv2.line(img, (mid, y), (mid, y + h), (255, 255, 255), 2)

    half = w // 2
    if v >= 0:
        fill = int(half * v)
        cv2.rectangle(img, (mid, y), (mid + fill, y + h), (0, 255, 0), -1)
    else:
        fill = int(half * (-v))
        cv2.rectangle(img, (mid - fill, y), (mid, y + h), (0, 0, 255), -1)

    cv2.putText(img, f"{label}: {v:+.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def safe_move(group, throttle, steer):
    if group is None:
        return
    t = -throttle if MOTOR_THROTTLE_INVERT else throttle
    group.move(t * THROTTLE_SCALE, steer * STEER_SCALE)

def compute_controls_for_person(kxy, kcf):
    ls = get_kpt(kxy, kcf, "left_shoulder")
    rs = get_kpt(kxy, kcf, "right_shoulder")
    lw = get_kpt(kxy, kcf, "left_wrist")
    rw = get_kpt(kxy, kcf, "right_wrist")
    lh = get_kpt(kxy, kcf, "left_hip")
    rh = get_kpt(kxy, kcf, "right_hip")

    if ls is None or rs is None:
        return False, 0.0, 0.0, None, 0.0, 0.0

    if ARM_SWAP_MODE:
        ls, rs = rs, ls
        lh, rh = rh, lh
        lw, rw = rw, lw

    shoulder_c = (ls + rs) * 0.5
    shoulder_w = dist(ls, rs)
    if shoulder_w < 1e-6:
        return False, 0.0, 0.0, None, 0.0, 0.0

    if lh is not None and rh is not None:
        hip_c = (lh + rh) * 0.5
        torso_c = (shoulder_c + hip_c) * 0.5
        lean_x = float((shoulder_c[0] - hip_c[0]) / shoulder_w)
        steer = clamp(lean_x * STEER_GAIN, -1.0, 1.0)
        steer = deadzone(steer, STEER_DEADZONE)
    else:
        torso_c = shoulder_c.copy()
        steer = 0.0

    if STEER_INVERT:
        steer = -steer

    right_ext01 = 0.0
    left_ext01 = 0.0
    if rw is not None:
        right_norm = dist(torso_c, rw) / shoulder_w
        right_ext01 = norm01(right_norm, EXT_MIN, EXT_MAX)
    if lw is not None:
        left_norm = dist(torso_c, lw) / shoulder_w
        left_ext01 = norm01(left_norm, EXT_MIN, EXT_MAX)

    raw = (right_ext01 - left_ext01)
    if THROTTLE_INVERT:
        raw = -raw

    throttle = clamp(raw * THROTTLE_GAIN, -1.0, 1.0)
    throttle = deadzone(throttle, THROTTLE_DEADZONE)

    return True, throttle, steer, torso_c.copy(), right_ext01, left_ext01

def safe_read_frame(cap):
    """Always returns (ok, frame) where frame is not None if ok=True."""
    if not DROP_OLD_FRAMES:
        ret, frame = cap.read()
        return (bool(ret and frame is not None), frame)

    # read one, then grab/drop a few, then retrieve last
    ret, frame = cap.read()
    if not ret or frame is None:
        return False, None

    for _ in range(GRAB_COUNT):
        cap.grab()

    ret2, frame2 = cap.retrieve()
    if ret2 and frame2 is not None:
        return True, frame2
    return True, frame  # fallback to the first good frame

def assign_players_stable(people, prev_p1x, prev_p2x, W):
    """
    people: list of dicts with center_x (display coords).
    Returns p1, p2. Uses nearest-to-previous assignment, with fallback left/right.
    """
    if not people:
        return None, None

    # left/right fallback
    people_lr = sorted(people, key=lambda p: p["center_x"])
    if len(people_lr) == 1:
        return people_lr[0], None

    # if no history, just left/right
    if prev_p1x is None and prev_p2x is None:
        return people_lr[0], people_lr[1]

    # compute normalized distances to previous x positions
    def ndist(xa, xb):
        return abs(xa - xb) / max(1.0, float(W))

    maxd = MATCH_MAX_DIST

    # candidate pairs
    a, b = people_lr[0], people_lr[1]
    ax, bx = a["center_x"], b["center_x"]

    # try two assignments: (a->p1,b->p2) vs (a->p2,b->p1)
    best = None
    best_cost = 1e9

    for p1cand, p2cand in [(a, b), (b, a)]:
        cost = 0.0
        ok = True
        if prev_p1x is not None:
            d1 = ndist(p1cand["center_x"], prev_p1x)
            if d1 > maxd:
                ok = False
            cost += d1
        if prev_p2x is not None:
            d2 = ndist(p2cand["center_x"], prev_p2x)
            if d2 > maxd:
                ok = False
            cost += d2
        if ok and cost < best_cost:
            best_cost = cost
            best = (p1cand, p2cand)

    if best is not None:
        return best[0], best[1]

    # fallback: keep left/right
    return people_lr[0], people_lr[1]


# ================= MAIN =================
model = YOLO("yolo11n-pose.pt")
try:
    model.fuse()
except Exception:
    pass

cap = open_camera_macos()
assert cap.isOpened(), "Could not open camera (check macOS Camera permissions)."

smooth = {
    "p1": {"throttle": 0.0, "steer": 0.0, "r": 0.0, "l": 0.0, "last_seen": 0.0, "last_x": None},
    "p2": {"throttle": 0.0, "steer": 0.0, "r": 0.0, "l": 0.0, "last_seen": 0.0, "last_x": None},
}

frame_idx = 0
last_results = None
last_motor_time = 0.0
motor_period = 1.0 / MOTOR_HZ

fps = 0.0
infer_fps = 0.0
last_fps_t = time.time()
frames_count = 0
infer_count = 0

print("\n2-Player mode (M3-stable):")
print("- Player 1 = left side of screen")
print("- Player 2 = right side of screen")
print(f"MIRROR_VIEW(display-only)={MIRROR_VIEW}  DEVICE={DEVICE}")
print(f"INFER={INFER_W}x{INFER_H} every {INFER_EVERY_N} frames")
print("Press 'q' in the OpenCV window to quit.\n")

prev_time = time.time()

while True:
    ok, frame_raw = safe_read_frame(cap)
    if not ok:
        print("ERROR: no frames received.")
        break

    now = time.time()
    dt = now - prev_time
    prev_time = now

    frame_infer = frame_raw

    if MIRROR_VIEW:
        annotated = cv2.flip(frame_raw, 1).copy()
    else:
        annotated = frame_raw.copy()

    H, W = annotated.shape[:2]
    small = cv2.resize(frame_infer, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)

    do_infer = (frame_idx % INFER_EVERY_N == 0)
    if do_infer:
        try:
            res = model.predict(
                small,
                conf=CONF_THRES,
                iou=0.45,
                classes=[0],
                verbose=False,
                device=DEVICE,
                max_det=10,
                imgsz=640,
                half=USE_HALF
            )
            last_results = res[0] if res else None
        except Exception as e:
            # Don't crash; just skip this inference
            last_results = None
            cv2.putText(annotated, f"Infer error: {type(e).__name__}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        infer_count += 1

    frame_idx += 1

    # FPS update
    frames_count += 1
    if now - last_fps_t >= 0.5:
        fps = frames_count / (now - last_fps_t)
        infer_fps = infer_count / (now - last_fps_t)
        frames_count = 0
        infer_count = 0
        last_fps_t = now

    people = []
    if last_results is not None:
        r = last_results
        sx = W / float(INFER_W)
        sy = H / float(INFER_H)

        if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
            kpts_xy_all = r.keypoints.xy.cpu().numpy()
            kpts_conf_all = r.keypoints.conf.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()

            for i in range(len(boxes)):
                ok2, thr, st, center_c, rext, lext = compute_controls_for_person(
                    kpts_xy_all[i], kpts_conf_all[i]
                )
                if not ok2 or center_c is None:
                    continue

                cx = float(center_c[0] * sx)
                cy = float(center_c[1] * sy)
                if MIRROR_VIEW:
                    cx = (W - 1) - cx

                area = bbox_area(boxes[i])
                people.append({
                    "center_x": cx,
                    "center_y": cy,
                    "throttle": float(thr),
                    "steer": float(st),
                    "r_ext": float(rext),
                    "l_ext": float(lext),
                    "area": float(area),
                    "box": boxes[i],
                    "sx": sx,
                    "sy": sy,
                })

    # keep 2 biggest
    people.sort(key=lambda p: -p["area"])
    people = people[:2]

    # stable assignment
    p1, p2 = assign_players_stable(
        people,
        smooth["p1"]["last_x"],
        smooth["p2"]["last_x"],
        W
    )

    def apply_smooth(dst, src, dt):
        if src is None:
            age = now - dst["last_seen"]
            tau = 0.07 if age > POSE_TIMEOUT_S else SMOOTH_TAU
            dst["throttle"] = smooth_step(dst["throttle"], 0.0, dt, tau)
            dst["steer"] = smooth_step(dst["steer"], 0.0, dt, tau)
            dst["r"] = smooth_step(dst["r"], 0.0, dt, tau)
            dst["l"] = smooth_step(dst["l"], 0.0, dt, tau)
        else:
            dst["last_seen"] = now
            dst["last_x"] = src["center_x"]
            dst["throttle"] = smooth_step(dst["throttle"], src["throttle"], dt, SMOOTH_TAU)
            dst["steer"] = smooth_step(dst["steer"], src["steer"], dt, SMOOTH_TAU)
            dst["r"] = smooth_step(dst["r"], src["r_ext"], dt, SMOOTH_TAU)
            dst["l"] = smooth_step(dst["l"], src["l_ext"], dt, SMOOTH_TAU)

    apply_smooth(smooth["p1"], p1, dt)
    apply_smooth(smooth["p2"], p2, dt)

    # motors
    if now - last_motor_time >= motor_period:
        last_motor_time = now
        safe_move(motor_group_1, smooth["p1"]["throttle"], smooth["p1"]["steer"])
        if ENABLE_BOT2:
            safe_move(motor_group_2, smooth["p2"]["throttle"], smooth["p2"]["steer"])

    # ===== HUD =====
    cv2.line(annotated, (W // 2, 0), (W // 2, H), (80, 80, 80), 2)

    pad = 20
    bar_w = 320
    bar_h = 20

    draw_centered_bar(annotated, pad, pad + 45, bar_w, bar_h, smooth["p1"]["throttle"], "P1 THROTTLE")
    draw_centered_bar(annotated, pad, pad + 85, bar_w, bar_h, smooth["p1"]["steer"], "P1 STEER")
    cv2.putText(annotated, f"P1 R:{smooth['p1']['r']:.2f} L:{smooth['p1']['l']:.2f}",
                (pad, pad + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    draw_centered_bar(annotated, W - pad - bar_w, pad + 45, bar_w, bar_h, smooth["p2"]["throttle"], "P2 THROTTLE")
    draw_centered_bar(annotated, W - pad - bar_w, pad + 85, bar_w, bar_h, smooth["p2"]["steer"], "P2 STEER")
    cv2.putText(annotated, f"P2 R:{smooth['p2']['r']:.2f} L:{smooth['p2']['l']:.2f}",
                (W - pad - bar_w, pad + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(annotated, f"FPS:{fps:.1f}  INFER_FPS:{infer_fps:.1f}  infer:{INFER_W}x{INFER_H} every {INFER_EVERY_N}",
                (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # draw boxes
    for person in people:
        x1, y1, x2, y2 = map(float, person["box"])
        sx = person["sx"]; sy = person["sy"]
        x1 *= sx; x2 *= sx
        y1 *= sy; y2 *= sy

        if MIRROR_VIEW:
            x1m = (W - 1) - x2
            x2m = (W - 1) - x1
            x1, x2 = x1m, x2m

        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

        label = "P1" if (p1 is person) else "P2" if (p2 is person) else "P?"
        cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (255, 255, 255), 2)
        cv2.putText(annotated, label, (x1i, max(25, y1i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("2 Player Real Steel", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

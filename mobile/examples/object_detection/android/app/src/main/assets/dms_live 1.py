

"""
Driver Monitoring System v2 — Improved Accuracy
Jetson Orin Nano | JetPack 6.2 | CUDA 12.6

Face Detection:  OpenCV YuNet (built-in CUDA OpenCV) → MediaPipe Landmarks
Object Detection: CLAHE preprocessing + dual YOLO + proper NMS
"""

import cv2
import numpy as np
import torch
import time
import mediapipe as mp
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
CAMERA_INDEX       = 0
FRAME_W            = 1280
FRAME_H            = 720
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Face detection (YuNet) ────────────────────────────────────
FACE_CONF          = 0.85    # YuNet face confidence — raise to reduce false detections
FACE_NMS           = 0.30    # YuNet NMS threshold

# ── EAR / MAR ─────────────────────────────────────────────────
EAR_THRESHOLD      = 0.25
EAR_CONSEC_FRAMES  = 5
MAR_THRESHOLD      = 0.50
MAR_CONSEC_FRAMES  = 7

# ── Head pose ─────────────────────────────────────────────────
YAW_LEFT_THRESH    = -20
YAW_RIGHT_THRESH   =  20
PITCH_UP_THRESH    = -15
PITCH_DOWN_THRESH  =  25
HEAD_CONSEC_FRAMES =  30
YAW_OFFSET         =  10

# ── YOLO ──────────────────────────────────────────────────────
YOLO_CONF          = 0.30    # Lowered for better phone detection sensitivity
YOLO_NMS           = 0.45    # IoU threshold for NMS
YOLO_INTERVAL      = 2       # Every 2nd frame (better than 3 for accuracy)

# ══════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ══════════════════════════════════════════════════════════════
LEFT_EYE            = [362, 385, 387, 263, 373, 380]  # p1=corner,p2-p3=upper,p4=corner,p5-p6=lower
RIGHT_EYE           = [33,  160, 158, 133, 153, 144]  # p1=corner,p2-p3=upper,p4=corner,p5-p6=lower
MOUTH               = [78, 308, 82, 87, 13, 14, 312, 317]  # Better lip landmarks for yawn accuracy
HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]

MODEL_POINTS = np.array([
    ( 0.0,    0.0,    0.0),
    ( 0.0, -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    ( 225.0, 170.0, -135.0),
    (-150.0,-150.0, -125.0),
    ( 150.0,-150.0, -125.0),
], dtype=np.float64)

CHAITANYA_CLASSES = ['Cigarette','Drinking','Eating','Phone','Seatbelt']
SOHAM_CLASSES     = ['Distracted','Drinking','Drowsy','Eating',
                     'PhoneUse','SafeDriving','Seatbelt','Smoking']
YOLO_WANT         = {'Phone','PhoneUse','Seatbelt','Cigarette','Smoking'}

RED    = (0,   0,   255)
ORANGE = (0,  140,  255)
GREEN  = (0,  220,   80)
YELLOW = (0,  220,  220)
WHITE  = (255, 255, 255)
BLACK  = (0,    0,    0)
DARK   = (20,  20,   20)

# ══════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING — CLAHE
#  Improves YOLO detection in poor lighting conditions
# ══════════════════════════════════════════════════════════════
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_frame(frame):
    """Apply CLAHE contrast enhancement for better detection in low light."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
def load_models():
    print(f"\n{'='*56}")
    print(f"  Driver Monitoring System v2 — Jetson Orin Nano")
    print(f"  Device : {DEVICE.upper()} | JetPack 6.2 | CUDA 12.6")
    print(f"{'='*56}")

    # ── YuNet Face Detector (built into OpenCV) ────────────────
    print("  [1/4] Loading YuNet Face Detector (OpenCV built-in)...")
    face_detector = cv2.FaceDetectorYN.create(
        model="face_detection_yunet_2023mar.onnx",
        config="",
        input_size=(FRAME_W, FRAME_H),
        score_threshold=FACE_CONF,
        nms_threshold=FACE_NMS,
        top_k=1,              # Only need the best face (driver)
        backend_id=cv2.dnn.DNN_BACKEND_CUDA,
        target_id=cv2.dnn.DNN_TARGET_CUDA
    )
    print("  [2/4] Loading MediaPipe Face Mesh (478 landmarks)...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,   # Raised from 0.6
        min_tracking_confidence=0.7     # Raised from 0.6
    )

    print("  [3/4] Loading YOLOv8n  (Chaitanya — 5 classes)...")
    m1 = YOLO('models/chaitanya/best.pt')
    m1.to(DEVICE)

    print("  [4/4] Loading YOLO11n  (Soham — 8 classes)...")
    m2 = YOLO('models/soham/best.pt')
    m2.to(DEVICE)

    print(f"\n  ✅  All models ready on {DEVICE.upper()}")
    print(f"{'='*56}\n")
    return face_detector, face_mesh, m1, m2

# ══════════════════════════════════════════════════════════════
#  YUNET FACE DETECTION
# ══════════════════════════════════════════════════════════════
def detect_face_yunet(frame, face_detector):
    """
    Returns (face_detected, face_bbox) where bbox = (x,y,w,h) or None.
    YuNet is a lightweight but accurate face detector built into OpenCV.
    """
    _, faces = face_detector.detect(frame)
    if faces is None or len(faces) == 0:
        return False, None
    # Take the LARGEST face (driver is closest = biggest bbox area)
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = map(int, best[:4])
    # Clamp to frame bounds
    x, y = max(0, x), max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    return True, (x, y, w, h)

# ══════════════════════════════════════════════════════════════
#  EAR / MAR
# ══════════════════════════════════════════════════════════════
def lm_pts(landmarks, indices, w, h):
    return np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
        dtype=np.float64
    )

def eye_aspect_ratio(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(pts):
    A = np.linalg.norm(pts[2] - pts[6])
    B = np.linalg.norm(pts[3] - pts[7])
    C = np.linalg.norm(pts[4] - pts[5])
    D = np.linalg.norm(pts[0] - pts[1])
    return (A + B + C) / (3.0 * D)

# ══════════════════════════════════════════════════════════════
#  HEAD POSE
# ══════════════════════════════════════════════════════════════
def estimate_head_pose(landmarks, w, h):
    img_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h)
         for i in HEAD_POSE_LANDMARKS],
        dtype=np.float64
    )
    f = w
    cam = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, img_pts, cam,
        np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0]*360, angles[1]*360

def classify_head_direction(pitch, yaw):
    yaw_c = yaw - YAW_OFFSET
    if yaw_c < YAW_LEFT_THRESH:    return "Looking LEFT  <"
    if yaw_c > YAW_RIGHT_THRESH:   return "Looking RIGHT >"
    if pitch  < PITCH_UP_THRESH:   return "Looking UP    ^"
    if pitch  > PITCH_DOWN_THRESH: return "Looking DOWN  v"
    return None

# ══════════════════════════════════════════════════════════════
#  YOLO with proper NMS between models
# ══════════════════════════════════════════════════════════════
def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def nms_detections(dets, iou_thresh=YOLO_NMS):
    """Apply NMS across detections from both models."""
    if not dets:
        return []
    # Group by unified label
    by_label = {}
    label_map = {
        'Phone': 'Phone', 'PhoneUse': 'Phone',
        'Seatbelt': 'Seatbelt',
        'Cigarette': 'Smoking', 'Smoking': 'Smoking',
    }
    for d in dets:
        unified = label_map.get(d['label'], d['label'])
        by_label.setdefault(unified, []).append({**d, 'unified': unified})

    final = []
    for label, boxes in by_label.items():
        boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
        kept = []
        while boxes:
            best = boxes.pop(0)
            kept.append(best)
            boxes = [b for b in boxes
                     if iou(best['bbox'], b['bbox']) < iou_thresh]
        final.extend(kept)
    return final

def run_yolo_models(frame, m1, m2):
    """Run both YOLO models on enhanced frame with proper NMS."""
    enhanced = enhance_frame(frame)
    raw = []

    for box in m1(enhanced, conf=YOLO_CONF,
                  verbose=False, device=DEVICE)[0].boxes:
        cls = CHAITANYA_CLASSES[int(box.cls[0])]
        if cls in YOLO_WANT:
            raw.append({'label': cls,
                        'conf':  float(box.conf[0]),
                        'bbox':  list(map(int, box.xyxy[0].cpu().numpy()))})

    for box in m2(enhanced, conf=YOLO_CONF,
                  verbose=False, device=DEVICE)[0].boxes:
        cls = SOHAM_CLASSES[int(box.cls[0])]
        if cls in YOLO_WANT:
            raw.append({'label': cls,
                        'conf':  float(box.conf[0]),
                        'bbox':  list(map(int, box.xyxy[0].cpu().numpy()))})

    return nms_detections(raw)

# ══════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════
def draw_text_bg(frame, text, pos, font_scale=0.65,
                 color=WHITE, bg=DARK, thickness=2, pad=6):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame,
                  (x-pad, y-th-pad), (x+tw+pad, y+pad), bg, -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_yolo_boxes(frame, dets):
    color_map = {
        'Phone':    (RED,    "PHONE"),
        'Seatbelt': (GREEN,  "SEATBELT"),
        'Smoking':  (ORANGE, "SMOKING"),
    }
    for d in dets:
        x1,y1,x2,y2 = d['bbox']
        unified = d.get('unified', d['label'])
        col, lbl = color_map.get(unified, (YELLOW, unified))
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        tag = f"{lbl} {d['conf']:.2f}"
        (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+4,y1), col, -1)
        cv2.putText(frame, tag, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2)

def draw_landmark_dots(frame, lm, indices, w, h, col):
    for i in indices:
        cv2.circle(frame,
                   (int(lm[i].x*w), int(lm[i].y*h)), 2, col, -1)

# ══════════════════════════════════════════════════════════════
#  DOWNLOAD YUNET MODEL IF MISSING
# ══════════════════════════════════════════════════════════════
def ensure_yunet():
    import os, urllib.request
    model_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(model_path):
        print("  Downloading YuNet face detection model...")
        url = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
               "face_detection_yunet/face_detection_yunet_2023mar.onnx")
        urllib.request.urlretrieve(url, model_path)
        print(f"  Downloaded: {model_path}")
    return model_path

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    ensure_yunet()
    face_detector, face_mesh, m1, m2 = load_models()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        return

    WIN = "Driver Monitoring System v2"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN,
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    ear_cnt = mar_cnt = head_cnt = 0
    yolo_dets   = []
    frame_cnt   = 0
    fps = 0.0
    t0  = time.time()
    no_face_cnt = 0

    print("Controls:  Q = Quit  |  S = Snapshot  |  F = Toggle Fullscreen")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_cnt += 1
        h, w = frame.shape[:2]

        # ── Step 1: YuNet face detection ─────────────────────
        face_found, face_bbox = detect_face_yunet(frame, face_detector)

        # Draw face bounding box from YuNet
        if face_found and face_bbox:
            fx, fy, fw, fh = face_bbox
            cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (180,180,0), 1)

        # ── Step 2: MediaPipe landmarks (only if face found) ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        alerts = []
        ear_v = mar_v = pitch = yaw = 0.0
        mp_face_ok = False

        if result.multi_face_landmarks and face_found:
            mp_face_ok = True
            no_face_cnt = 0
            lm = result.multi_face_landmarks[0].landmark

            draw_landmark_dots(frame, lm, LEFT_EYE,  w, h, (0,255,180))
            draw_landmark_dots(frame, lm, RIGHT_EYE, w, h, (0,255,180))
            draw_landmark_dots(frame, lm, MOUTH,     w, h, (180,220,0))

            # EAR — both eyes must be closed independently
            le      = lm_pts(lm, LEFT_EYE,  w, h)
            re      = lm_pts(lm, RIGHT_EYE, w, h)
            ear_l   = eye_aspect_ratio(le)
            ear_r   = eye_aspect_ratio(re)
            ear_v   = (ear_l + ear_r) / 2.0
            # Only trigger if BOTH eyes are below threshold (avoids wink false positives)
            both_closed = ear_l < EAR_THRESHOLD and ear_r < EAR_THRESHOLD
            ear_cnt = ear_cnt+1 if both_closed else 0
            if ear_cnt >= EAR_CONSEC_FRAMES:
                alerts.append(("EYES CLOSED  WAKE UP!", RED))

            # MAR
            mp_pts = lm_pts(lm, MOUTH, w, h)
            mar_v  = mouth_aspect_ratio(mp_pts)
            mar_cnt = mar_cnt+1 if mar_v > MAR_THRESHOLD else 0
            if mar_cnt >= MAR_CONSEC_FRAMES:
                alerts.append(("YAWNING DETECTED!", ORANGE))

            # Head pose
            pitch, yaw = estimate_head_pose(lm, w, h)
            hdir = classify_head_direction(pitch, yaw)
            head_cnt = head_cnt+1 if hdir else 0
            if head_cnt >= HEAD_CONSEC_FRAMES:
                alerts.append((f"DISTRACTED: {hdir}", RED))

        elif not face_found:
            no_face_cnt += 1
            ear_cnt = mar_cnt = head_cnt = 0
            if no_face_cnt > 10:
                alerts.append(("NO FACE DETECTED", ORANGE))

        # ── Step 3: YOLO ──────────────────────────────────────
        if frame_cnt % YOLO_INTERVAL == 0:
            yolo_dets = run_yolo_models(frame, m1, m2)

        draw_yolo_boxes(frame, yolo_dets)

        unified_labels = {d.get('unified', d['label']) for d in yolo_dets}
        has_seatbelt = 'Seatbelt' in unified_labels
        has_phone    = 'Phone'    in unified_labels
        has_smoking  = 'Smoking'  in unified_labels

        if has_phone:    alerts.append(("PUT DOWN YOUR PHONE!", RED))
        if has_smoking:  alerts.append(("NO SMOKING!", ORANGE))
        if not has_seatbelt:
            alerts.append(("FASTEN YOUR SEATBELT!", RED))

        # ── FPS ───────────────────────────────────────────────
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps = frame_cnt / elapsed
            frame_cnt = 0
            t0 = time.time()

        # ══════════════════════════════════════════════════════
        #  HUD
        # ══════════════════════════════════════════════════════
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (330,115), DARK, -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        face_status = "Face: YuNet+MP OK" if mp_face_ok else ("Face: YuNet only" if face_found else "Face: NOT FOUND")
        face_col    = GREEN if mp_face_ok else (YELLOW if face_found else RED)
        cv2.putText(frame, f"FPS {fps:.1f}  |  {face_status}",
                    (10,22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, face_col, 1)
        cv2.putText(frame,
                    f"EAR {ear_v:.3f} (thr {EAR_THRESHOLD})",
                    (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    RED if ear_v < EAR_THRESHOLD else GREEN, 1)
        cv2.putText(frame,
                    f"MAR {mar_v:.3f} (thr {MAR_THRESHOLD})",
                    (10,64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    ORANGE if mar_v > MAR_THRESHOLD else GREEN, 1)
        cv2.putText(frame,
                    f"Pitch {pitch:.1f}  Yaw {yaw:.1f}",
                    (10,84), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, YELLOW, 1)
        cv2.putText(frame, f"YOLO objs: {len(yolo_dets)}",
                    (10,104), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, YELLOW, 1)

        # Top-right: head direction
        hdir_now = classify_head_direction(pitch, yaw)
        hdir_txt = hdir_now if hdir_now else "Forward  OK"
        hdir_col = RED      if hdir_now else GREEN
        (dtw,_),_ = cv2.getTextSize(
            hdir_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        draw_text_bg(frame, hdir_txt,
                     (w-dtw-20, 35), font_scale=0.7, color=hdir_col)

        # Top-right: seatbelt
        sb_txt = "SEATBELT ON" if has_seatbelt else "NO SEATBELT"
        sb_col = GREEN         if has_seatbelt else RED
        (stw,_),_ = cv2.getTextSize(
            sb_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        draw_text_bg(frame, sb_txt,
                     (w-stw-20, 70), font_scale=0.65, color=sb_col)

        # Centre-bottom alerts
        if alerts:
            ay = h - len(alerts)*50 - 20
            for (atxt, acol) in alerts:
                (atw,ath),_ = cv2.getTextSize(
                    atxt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                bx = (w-atw)//2
                ov2 = frame.copy()
                cv2.rectangle(ov2, (bx-12,ay-ath-10),
                              (bx+atw+12,ay+8), BLACK, -1)
                cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)
                cv2.rectangle(frame, (bx-12,ay-ath-10),
                              (bx+atw+12,ay+8), acol, 1)
                cv2.putText(frame, atxt, (bx, ay),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, acol, 2)
                ay += 50
        else:
            safe = "SAFE DRIVING"
            (sw,_),_ = cv2.getTextSize(
                safe, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            draw_text_bg(frame, safe,
                         ((w-sw)//2, h-20),
                         font_scale=0.9, color=GREEN)

        cv2.putText(frame,
                    "YuNet+MediaPipe | YOLOv8n+YOLO11n+CLAHE | Jetson GPU",
                    (8, h-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (90,90,90), 1)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved: {fname}")
        elif key == ord('f'):
            prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
                if prop == cv2.WINDOW_FULLSCREEN
                else cv2.WINDOW_FULLSCREEN)

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    main()
ENDOFFILE


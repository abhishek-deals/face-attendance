# 03_attendance.py — Real-Time Face Recognition & Attendance
#
# ══════════════════════════════════════════════════════════
#  PERFORMANCE OPTIMIZATIONS (AMD Athlon / 4GB RAM)
# ══════════════════════════════════════════════════════════
#
#  1. FRAME SKIPPING   — Process only 1 of every 3 frames.
#                        Cuts CPU usage by ~66%.
#
#  2. LOW RESOLUTION   — Camera forced to 320x240.
#                        ~4x less data per frame vs 640x480.
#
#  3. LOW FPS          — Camera capped at 15 FPS.
#                        Reduces heat vs default 30 FPS.
#
#  4. GRAYSCALE ONLY   — Convert to grayscale before any
#                        face detection. 1/3 the memory.
#
#  5. BOUNDED FACE SIZE — minSize=(50,50) maxSize=(200,200)
#                         Ignores too-small/large regions.
#
#  6. SIMPLE DISPLAY   — No fancy overlays or animations.
#                         Just text. Minimal GPU draw calls.
#
#  7. MEMORY CLEANUP   — cam.release() + destroyAllWindows()
#                         always called on exit.
# ══════════════════════════════════════════════════════════

import cv2
import os
import sys
from datetime import datetime
from db import setup_db, mark_attendance, export_csv
# NOTE: numpy is NOT imported — it is not needed here and
# wastes ~50MB RAM. opencv handles all array operations internally.


# ──────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# Adjust these if laptop runs hot or lags.
# ──────────────────────────────────────────────
SKIP_FRAMES      = 3    # Process 1 of every N frames (increase if hot)
CONFIDENCE_LIMIT = 70   # Lower = stricter. Range: 0 (perfect) to 100+
CAM_WIDTH        = 320
CAM_HEIGHT       = 240
CAM_FPS          = 15


# ──────────────────────────────────────────────
# STEP 1: Verify required files exist
# ──────────────────────────────────────────────
MODEL_PATH   = os.path.join("trainer", "model.yml")
NAMES_PATH   = os.path.join("trainer", "names.txt")
CASCADE_PATH = os.path.join("haarcascade",
               "haarcascade_frontalface_default.xml")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Trained model not found!")
    print(f"  Expected: {MODEL_PATH}")
    print("  Run 02_train.py first.")
    sys.exit(1)

if not os.path.exists(NAMES_PATH):
    print("ERROR: Names file not found!")
    print(f"  Expected: {NAMES_PATH}")
    print("  Run 02_train.py first.")
    sys.exit(1)

if not os.path.exists(CASCADE_PATH):
    print("ERROR: Haarcascade XML not found!")
    print(f"  Expected: {CASCADE_PATH}")
    print("  Download haarcascade_frontalface_default.xml")
    print("  from https://github.com/opencv/opencv/tree/master/data/haarcascades")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 2: Load LBPH model
# ──────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create()

try:
    recognizer.read(MODEL_PATH)
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 3: Load face detector (haarcascade)
# ──────────────────────────────────────────────
detector = cv2.CascadeClassifier(CASCADE_PATH)

if detector.empty():
    print("ERROR: Failed to load haarcascade classifier.")
    print("The XML file may be corrupted. Re-download it.")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 4: Load ID → Name mapping from names.txt
# Format: "<int_id>:<name>"  one per line
# ──────────────────────────────────────────────
id_to_name = {}

try:
    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    id_to_name[int(parts[0])] = parts[1]
                except ValueError:
                    pass
except Exception as e:
    print(f"ERROR reading names.txt: {e}")
    sys.exit(1)

if len(id_to_name) == 0:
    print("ERROR: names.txt is empty or invalid.")
    print("Re-run 02_train.py to regenerate it.")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 5: Initialize database
# ──────────────────────────────────────────────
setup_db()


# ──────────────────────────────────────────────
# STEP 6: Open camera with LOW settings
# ──────────────────────────────────────────────
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("ERROR: Cannot open camera.")
    print("  Make sure camera is connected and not used by another app.")
    sys.exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cam.set(cv2.CAP_PROP_FPS,          CAM_FPS)


# ──────────────────────────────────────────────
# STEP 7: State variables
# ──────────────────────────────────────────────
marked_set   = set()    # Names marked this session
frame_count  = 0        # Total frames read (used for skip logic)

# Status bar displayed on every frame
status_msg   = "System Ready — Face the camera"
status_color = (255, 255, 255)  # White


print()
print("=" * 45)
print("   ATTENDANCE SYSTEM RUNNING")
print("=" * 45)
print(f"  Camera     : {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS} FPS")
print(f"  Frame skip : 1 of every {SKIP_FRAMES} frames processed")
print(f"  Confidence : faces above {CONFIDENCE_LIMIT} marked Unknown")
print(f"  Students   : {len(id_to_name)} loaded")
print()
print("  Press ESC to stop the session.")
print("=" * 45)
print()


# ──────────────────────────────────────────────
# STEP 8: Main recognition loop
# ──────────────────────────────────────────────
while True:
    ret, frame = cam.read()

    if not ret:
        print("  [!] Camera read failed. Stopping.")
        break

    frame_count += 1

    # ════════════════════════════════════════════
    # FRAME SKIP LOGIC
    # Only process every SKIP_FRAMES-th frame.
    # For skipped frames: just show last status + display.
    # This alone cuts CPU usage by 66% (SKIP_FRAMES=3).
    # ════════════════════════════════════════════
    if frame_count % SKIP_FRAMES != 0:
        # Display current status on un-processed frame
        cv2.putText(
            frame,
            status_msg,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            1
        )
        cv2.putText(
            frame,
            f"Marked: {len(marked_set)}",
            (5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )
        cv2.imshow("Attendance System  [ESC = Exit]", frame)

        if cv2.waitKey(1) == 27:
            break
        continue  # Skip to next frame without processing

    # ════════════════════════════════════════════
    # PROCESS THIS FRAME
    # ════════════════════════════════════════════

    # Convert to grayscale FIRST — never process color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # scaleFactor=1.3  → detection step size (1.1 = slower but catches more)
    # minNeighbors=5   → reject weak detections (reduces false positives)
    # minSize=(50,50)  → ignore tiny blobs
    # maxSize=(200,200)→ ignore objects too large to be a face at this res
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        maxSize=(200, 200)
    )

    for (x, y, w, h) in faces:
        # Extract and resize face region to 100x100
        # Must match the size used during training
        face_region  = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, (100, 100))

        # ── LBPH Prediction ──────────────────
        # sid        → predicted student integer ID
        # confidence → lower = more confident
        #              0.0 = perfect match
        #              70+  = likely unknown
        sid, confidence = recognizer.predict(face_resized)

        if confidence < CONFIDENCE_LIMIT:
            # ── Known person ─────────────────
            name = id_to_name.get(sid, "Unknown")

            # Mark attendance (returns True only on first mark today)
            is_new = mark_attendance(str(sid), name)

            if is_new:
                # Newly marked this session
                status_msg   = f"MARKED: {name}"
                status_color = (0, 255, 0)  # Green
                marked_set.add(name)
                print(f"  [OK] Marked: {name:<20s}  "
                      f"[Conf: {confidence:.1f}]  "
                      f"{datetime.now().strftime('%H:%M:%S')}")
            else:
                # Already marked earlier today
                # Truncate name if too long for 320px frame
                short_name = name[:14] if len(name) > 14 else name
                status_msg   = f"Already: {short_name}"
                status_color = (0, 255, 255)  # Yellow

            # Green rectangle + name label
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )
            cv2.putText(
                frame, name,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

        else:
            # ── Unknown / Unrecognized person ─
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 0, 255), 2
            )
            cv2.putText(
                frame, "Unknown",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1
            )
            status_msg   = "Unknown face detected"
            status_color = (0, 0, 255)  # Red

    # ── HUD overlay ──────────────────────────
    # Show status message at top-left of frame
    # Truncate to 32 chars max to fit 320px-wide frame
    display_msg = status_msg[:32] if len(status_msg) > 32 else status_msg
    cv2.putText(
        frame,
        display_msg,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        1
    )

    # Show count of students marked this session
    cv2.putText(
        frame,
        f"Marked: {len(marked_set)}",
        (5, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )

    cv2.imshow("Attendance System  [ESC = Exit]", frame)

    # waitKey(1) = 1ms wait — minimum needed for cv2.imshow to render
    if cv2.waitKey(1) == 27:  # ESC key
        break


# ──────────────────────────────────────────────
# STEP 9: Cleanup — always release camera & windows
# ──────────────────────────────────────────────
cam.release()
cv2.destroyAllWindows()

# Auto-save daily CSV after each session ends
# Saves to attendance/attendance_YYYY-MM-DD.csv
try:
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = export_csv(date=today)
    if csv_path:
        print(f"  Daily CSV saved: {csv_path}")
except Exception:
    pass  # Never crash on cleanup

# Final session summary
print()
print("=" * 45)
print("  Session Ended")
print(f"  Students marked this session: {len(marked_set)}")
if marked_set:
    for sname in sorted(marked_set):
        print(f"    [OK] {sname}")
print()
print("  To view/export report: python 04_report.py")
print("=" * 45)

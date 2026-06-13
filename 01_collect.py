# 01_collect.py — Face Image Collection Module
# Captures 50 face images per student for LBPH training.
#
# PERFORMANCE OPTIMIZATIONS FOR AMD ATHLON / 4GB RAM:
#   - Camera capped at 320x240 (lowest usable resolution)
#   - Camera FPS limited to 15 (reduces heat/CPU load)
#   - Grayscale conversion done IMMEDIATELY on each frame
#   - Face saved as 100x100 pixels only (tiny file size)
#   - 200ms delay between captures (CPU stays cool)
#   - Only 50 images collected (enough for LBPH accuracy)
# ──────────────────────────────────────────────────────

import cv2
import os
import sys
from db import setup_db, add_student


# ──────────────────────────────────────────────
# STEP 1: Initialize database
# ──────────────────────────────────────────────
setup_db()


# ──────────────────────────────────────────────
# STEP 2: Verify haarcascade XML exists
# Download from: https://github.com/opencv/opencv/
#   tree/master/data/haarcascades
# Place as: haarcascade/haarcascade_frontalface_default.xml
# ──────────────────────────────────────────────
CASCADE_PATH = os.path.join(
    "haarcascade", "haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_PATH):
    print("ERROR: haarcascade XML not found!")
    print(f"Expected at: {CASCADE_PATH}")
    print()
    print("Download it from:")
    print("https://github.com/opencv/opencv/tree/master/data/haarcascades")
    print("File: haarcascade_frontalface_default.xml")
    print("Place it inside the 'haarcascade/' folder.")
    sys.exit(1)

# Load the face detector
detector = cv2.CascadeClassifier(CASCADE_PATH)

if detector.empty():
    print("ERROR: Failed to load haarcascade classifier.")
    print("The XML file may be corrupted. Re-download it.")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 3: Get student information from user
# ──────────────────────────────────────────────
print("=" * 40)
print("   FACE COLLECTION - ADD STUDENT")
print("=" * 40)
print()

# --- Student ID input (numeric only) ---
while True:
    sid = input("Enter Student ID (numbers only): ").strip()

    if sid == "":
        print("  [!] Student ID cannot be empty. Try again.")
        continue

    if not sid.isdigit():
        print("  [!] Student ID must be numeric only (e.g., 101). Try again.")
        continue

    # Valid numeric ID
    break

# --- Student Name input ---
while True:
    name = input("Enter Student Name: ").strip()

    if name == "":
        print("  [!] Name cannot be empty. Try again.")
        continue

    if len(name) < 2:
        print("  [!] Name must be at least 2 characters. Try again.")
        continue

    # Remove characters that are invalid in folder names
    safe_name = "".join(
        c for c in name if c.isalnum() or c in (" ", "-", "_")
    ).strip()

    if safe_name == "":
        print("  [!] Name contains only invalid characters. Try again.")
        continue

    # Replace spaces with underscores for safe folder names on Windows
    name = safe_name.replace(" ", "_")
    break

print()
print(f"  Student ID : {sid}")
print(f"  Student Name: {name}")
print()


# ──────────────────────────────────────────────
# STEP 4: Create save folder
# Folder pattern: dataset/<sid>_<name>
# ──────────────────────────────────────────────
save_path = os.path.join("dataset", f"{sid}_{name}")
os.makedirs(save_path, exist_ok=True)
print(f"  Saving images to: {save_path}")


# ──────────────────────────────────────────────
# STEP 5: Open camera with LOW settings
# 320x240 resolution = ~4x less data than 640x480
# 15 FPS = less heat compared to 30 FPS
# ──────────────────────────────────────────────
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print()
    print("ERROR: Cannot open camera (index 0).")
    print("Check if camera is connected or not in use by another app.")
    sys.exit(1)

# Force low resolution to reduce CPU/RAM load
cam.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cam.set(cv2.CAP_PROP_FPS, 15)      # Limit FPS to reduce heat

print()
print("  Camera opened at 320x240 @ 15 FPS")
print("  Look at the camera.")
print("  Press ESC to stop early.")
print()


# ──────────────────────────────────────────────
# STEP 6: Image collection loop
# ──────────────────────────────────────────────
count = 0
TARGET = 50  # 50 images is plenty for LBPH accuracy

while count < TARGET:
    ret, frame = cam.read()

    if not ret:
        print("  [!] Failed to read from camera. Stopping.")
        break

    # ── Convert to grayscale IMMEDIATELY ──────
    # Never process BGR color frames.
    # Grayscale = 1/3 the data = faster everything.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect faces in grayscale frame ───────
    # scaleFactor=1.3  → smaller value = more thorough but slower
    # minNeighbors=5   → higher = fewer false detections
    # minSize=(60,60)  → ignore tiny faces (noise reduction)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Stop collecting if we already have enough images
        if count >= TARGET:
            break

        count += 1

        # Save only the face region, resized to 100x100
        # 100x100 is the smallest size that preserves LBPH features
        face_crop = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, (100, 100))

        img_file = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_file, face_resized)

        # Draw green rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show progress counter on frame
    cv2.putText(
        frame,
        f"Captured: {count}/{TARGET}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collecting Faces  [ESC = Stop]", frame)

    # ── 200ms delay between frames ────────────
    # This is intentional to prevent CPU spike.
    # At 15 FPS we already have ~67ms between frames,
    # but waitKey(200) ensures we never over-process.
    key = cv2.waitKey(200)
    if key == 27:  # ESC key
        print("  [!] Collection stopped early by user.")
        break


# ──────────────────────────────────────────────
# STEP 7: Cleanup and finalize
# ──────────────────────────────────────────────
cam.release()
cv2.destroyAllWindows()

if count == 0:
    print()
    print("ERROR: No face images were captured.")
    print("  - Make sure your face is clearly visible to the camera.")
    print("  - Ensure lighting is adequate.")
    print("  - Try moving closer to the camera.")
    sys.exit(1)

# Register student in database
add_student(sid, name)

print()
print("=" * 40)
print(f"  Collection COMPLETE!")
print(f"  Student : {name}  (ID: {sid})")
print(f"  Images  : {count} images saved")
print(f"  Location: {save_path}/")
print("=" * 40)
print()
print("To add more students: run 01_collect.py again.")
print("When all students are added, run:")
print("   python 02_train.py")

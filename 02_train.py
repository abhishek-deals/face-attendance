# 02_train.py — LBPH Model Training Module
#
# PERFORMANCE NOTE:
#   LBPH (Local Binary Patterns Histogram) is the LIGHTEST
#   face recognition algorithm available in OpenCV.
#   - No GPU required
#   - No deep learning weights
#   - 50 images per student → trains in ~5-15 seconds
#   - Model file is tiny (< 1MB for 20 students)
#   - RAM usage: ~150-300MB peak during training only
#
# After training, model is saved to trainer/model.yml
# and ID→name mapping is saved to trainer/names.txt
# ──────────────────────────────────────────────────────

import cv2
import numpy as np
import os
import sys
from PIL import Image


# ──────────────────────────────────────────────
# STEP 1: Verify dataset folder exists
# ──────────────────────────────────────────────
DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"

if not os.path.exists(DATASET_PATH):
    print("ERROR: 'dataset/' folder not found.")
    print("Run 01_collect.py first to collect face images.")
    sys.exit(1)

# Check if dataset has any student folders
subfolders = [
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
]

if len(subfolders) == 0:
    print("ERROR: 'dataset/' folder is empty.")
    print("Run 01_collect.py first to collect face images.")
    sys.exit(1)

# Ensure trainer output folder exists
os.makedirs(TRAINER_PATH, exist_ok=True)


# ──────────────────────────────────────────────
# STEP 2: Create LBPH recognizer
#
# LBPH Parameters (tuned for low-end hardware):
#   radius=1      → compare each pixel with its 1-pixel neighbor ring
#   neighbors=8   → 8 sample points on that ring
#   grid_x=8      → divide face into 8 horizontal cells
#   grid_y=8      → divide face into 8 vertical cells
#
# These defaults work well with 100x100 face images.
# Increasing them improves accuracy slightly but uses more RAM.
# ──────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)


# ──────────────────────────────────────────────
# FUNCTION: load_training_data(dataset_path)
#
# Walks through dataset/<sid>_<name>/ folders.
# Loads all .jpg images as grayscale numpy arrays.
# Returns:
#   face_samples  → list of 100x100 numpy arrays
#   ids           → list of integer student IDs (parallel to face_samples)
#   id_name_map   → dict {int_id: "Name"}
# ──────────────────────────────────────────────
def load_training_data(dataset_path):
    """Load all face images from dataset/ folder."""
    face_samples = []
    ids = []
    id_name_map = {}

    folder_list = sorted(os.listdir(dataset_path))

    for folder in folder_list:
        folder_path = os.path.join(dataset_path, folder)

        # Skip files — only process directories
        if not os.path.isdir(folder_path):
            continue

        # Folder name format: "<sid>_<name>"
        # Example: "101_John_Doe" → sid=101, name="John_Doe"
        parts = folder.split("_", 1)

        if len(parts) != 2:
            print(f"  [!] Skipping '{folder}' — unexpected folder name format.")
            print(f"      Expected format: <number>_<name>  (e.g. 101_Alice)")
            continue

        # Parse the student ID (must be an integer)
        try:
            student_id = int(parts[0])
            student_name = parts[1]
        except ValueError:
            print(f"  [!] Skipping '{folder}' — ID part is not a number.")
            continue

        id_name_map[student_id] = student_name

        # Load each image file in this student's folder
        img_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".jpg")
        ]

        loaded_count = 0

        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)

            try:
                # Open with PIL and convert to strict grayscale uint8
                # This is more reliable than cv2.imread for consistency
                pil_image = Image.open(img_path).convert("L")
                img_array = np.array(pil_image, dtype="uint8")

                face_samples.append(img_array)
                ids.append(student_id)
                loaded_count += 1

            except Exception as e:
                # Skip corrupt/unreadable images silently
                pass

        print(f"  Loaded: {student_name:<20s} — {loaded_count} images")

    return face_samples, ids, id_name_map


# ──────────────────────────────────────────────
# STEP 3: Load all training data
# ──────────────────────────────────────────────
print()
print("=" * 45)
print("   LBPH MODEL TRAINING")
print("=" * 45)
print()
print("Loading training images from dataset/...")
print()

faces, ids, id_name_map = load_training_data(DATASET_PATH)

# Validate that we actually got data
if len(faces) == 0:
    print()
    print("ERROR: No valid face images could be loaded.")
    print("  - Check that dataset/ folders contain .jpg files.")
    print("  - Folder name format must be: <number>_<name>")
    print("    Example: 101_Alice or 202_Bob_Smith")
    sys.exit(1)

if len(set(ids)) < 1:
    print("ERROR: Could not parse any student IDs.")
    sys.exit(1)

print()
print(f"  Students  : {len(id_name_map)}")
print(f"  Total imgs: {len(faces)}")
print(f"  Avg/student: {len(faces) // len(id_name_map)} images")
print()
print("Training LBPH model... Please wait.")
print("(This may take 5-20 seconds on your hardware)")
print()


# ──────────────────────────────────────────────
# STEP 4: Train the LBPH model
# ──────────────────────────────────────────────
try:
    recognizer.train(faces, np.array(ids))
except cv2.error as e:
    print(f"ERROR during training: {e}")
    print("Ensure all images in dataset/ are valid 100x100 grayscale JPEGs.")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 5: Save model to trainer/model.yml
# ──────────────────────────────────────────────
model_path = os.path.join(TRAINER_PATH, "model.yml")

try:
    recognizer.write(model_path)
    print(f"  Model saved : {model_path}")
except Exception as e:
    print(f"ERROR saving model: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 6: Save ID → Name mapping to trainer/names.txt
# Format per line: "<int_id>:<name>"
# Example:  "101:Alice"
# ──────────────────────────────────────────────
names_path = os.path.join(TRAINER_PATH, "names.txt")

try:
    with open(names_path, "w", encoding="utf-8") as f:
        for sid, sname in id_name_map.items():
            f.write(f"{sid}:{sname}\n")
    print(f"  Names saved : {names_path}")
except Exception as e:
    print(f"ERROR saving names file: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 7: Free memory immediately after training
# Training arrays can occupy significant RAM.
# del them to free before the script exits.
# ──────────────────────────────────────────────
del faces
del ids

print()
print("=" * 45)
print("  Training COMPLETE!")
print()
print("  Students trained:")
for sid, sname in id_name_map.items():
    print(f"    ID {sid:>4d} → {sname}")
print()
print("  Next step: run attendance system with:")
print("    python 03_attendance.py")
print("=" * 45)

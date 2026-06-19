# 02_train.py — LBPH Model Training Module
#
# ACCURACY BOOST: DATA AUGMENTATION
#   Only 8 real photos needed per student.
#   Each photo is automatically expanded into ~11 variants:
#     - 4 brightness levels  (darker/brighter)
#     - 4 small rotations    (±5°, ±10°)
#     - 1 horizontal flip
#     - 1 gaussian noise
#     - 1 original
#   = 8 photos × 11 = 88 training samples per student.
#   This gives the same accuracy as collecting 50+ raw photos.
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
    radius=2,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

# CLAHE: same equalizer used in 03_attendance.py
# MUST apply to training images too, or model is trained on
# different-looking images than what it sees at recognition time.
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ──────────────────────────────────────────────
# FUNCTION: augment_face(img_array)
#
# Takes one 100x100 grayscale face array and returns
# ~11 synthetic variants via brightness, rotation,
# flip, and noise transforms.
#
# WHY: LBPH accuracy improves greatly with more
# training samples. 8 real photos × 11 augmentations
# = 88 effective training samples — similar to
# collecting 50+ raw photos manually.
# ──────────────────────────────────────────────
def augment_face(img_array):
    """Generate augmented variants of a single grayscale face image."""
    variants = [img_array]  # 1. Always include original
    h, w = img_array.shape
    center = (w // 2, h // 2)

    # 2. Brightness: 4 levels (darker + brighter)
    for factor in [0.70, 0.85, 1.15, 1.30]:
        bright = np.clip(
            img_array.astype(np.float32) * factor, 0, 255
        ).astype(np.uint8)
        variants.append(bright)

    # 3. Small rotations: ±5°, ±10°
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img_array, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        variants.append(rotated)

    # 4. Horizontal flip
    variants.append(cv2.flip(img_array, 1))

    # 5. Gaussian noise
    noise = np.random.normal(0, 8, img_array.shape).astype(np.int16)
    noisy = np.clip(
        img_array.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    variants.append(noisy)

    return variants  # 11 total variants per photo


# ──────────────────────────────────────────────
# FUNCTION: load_training_data(dataset_path)
#
# Walks through dataset/<sid>_<name>/ folders.
# Loads all .jpg images as grayscale numpy arrays,
# then AUGMENTS each one into ~11 variants.
# Returns:
#   face_samples  → list of 100x100 numpy arrays
#   ids           → list of integer student IDs (parallel to face_samples)
#   id_name_map   → dict {int_id: "Name"}
# ──────────────────────────────────────────────
def load_training_data(dataset_path):
    """Load and augment all face images from dataset/ folder."""
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

        real_count = 0
        aug_count  = 0

        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)

            try:
                # Open with PIL — resize to 100x100 grayscale
                pil_image = Image.open(img_path).convert("L")
                # Ensure all images are 100×100 before augmentation
                pil_image = pil_image.resize((100, 100), Image.LANCZOS)
                img_array = np.array(pil_image, dtype="uint8")

                # Apply CLAHE equalization — MUST match 03_attendance.py
                # Recognition applies CLAHE before predict(), so training
                # must apply it too. Without this, model is trained on
                # raw images but recognizes equalized images → wrong matches.
                img_array = _clahe.apply(img_array)

                # Generate augmented variants
                for variant in augment_face(img_array):
                    face_samples.append(variant)
                    ids.append(student_id)
                    aug_count += 1

                real_count += 1

            except Exception:
                # Skip corrupt/unreadable images silently
                pass

        print(f"  Loaded: {student_name:<20s} - {real_count} photos "
              f"-> {aug_count} training samples (augmented)")

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
    print(f"    ID {sid:>4d} -> {sname}")
print()
print("  Next step: run attendance system with:")
print("    python 03_attendance.py")
print("=" * 45)

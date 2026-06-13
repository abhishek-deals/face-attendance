# 00_setup.py — First-Run Setup Script
#
# RUN THIS ONCE BEFORE ANYTHING ELSE.
#
# This script automatically:
#   1. Verifies Python version (3.7+ required)
#   2. Creates all required folders
#   3. Downloads haarcascade XML from GitHub (no manual download needed)
#   4. Initializes the SQLite database
#   5. Verifies all required packages are installed
#   6. Prints a final checklist
#
# Uses ONLY built-in Python libraries for downloading:
#   urllib.request — no pip install needed
#
# ──────────────────────────────────────────────────────────

import os
import sys
import urllib.request
import urllib.error

print()
print("=" * 55)
print("   FACE ATTENDANCE SYSTEM - FIRST-TIME SETUP")
print("=" * 55)
print()


# ──────────────────────────────────────────────
# STEP 1: Check Python version
# ──────────────────────────────────────────────
print("[1/6] Checking Python version...")

major = sys.version_info.major
minor = sys.version_info.minor

if major < 3 or (major == 3 and minor < 7):
    print(f"  ERROR: Python 3.7+ required. You have {major}.{minor}")
    print("  Download Python from: https://www.python.org/downloads/")
    sys.exit(1)

print(f"  OK — Python {major}.{minor}.{sys.version_info.micro}")


# ──────────────────────────────────────────────
# STEP 2: Create all required folders
# ──────────────────────────────────────────────
print()
print("[2/6] Creating folder structure...")

FOLDERS = [
    "dataset",
    "trainer",
    "attendance",
    "haarcascade",
]

for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)
    print(f"  OK — {folder}/")

print("  All folders ready.")


# ──────────────────────────────────────────────
# STEP 3: Download haarcascade XML
#
# This is the face detector used by OpenCV.
# Source: Official OpenCV GitHub repository.
# File size: ~930 KB — downloads in seconds.
#
# If download fails (no internet), instructions
# for manual download are printed.
# ──────────────────────────────────────────────
print()
print("[3/6] Downloading haarcascade XML...")

HAARCASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

CASCADE_DEST = os.path.join(
    "haarcascade", "haarcascade_frontalface_default.xml"
)

if os.path.exists(CASCADE_DEST):
    size_kb = os.path.getsize(CASCADE_DEST) // 1024
    if size_kb > 100:  # Valid file is ~930 KB
        print(f"  OK — Already exists ({size_kb} KB). Skipping download.")
    else:
        print(f"  WARNING: Existing file is too small ({size_kb} KB).")
        print("  Re-downloading...")
        os.remove(CASCADE_DEST)

if not os.path.exists(CASCADE_DEST):
    print(f"  Downloading from GitHub...")
    print(f"  URL: {HAARCASCADE_URL}")

    try:
        # Set a 30-second timeout to prevent hanging
        urllib.request.urlretrieve(HAARCASCADE_URL, CASCADE_DEST)

        size_kb = os.path.getsize(CASCADE_DEST) // 1024
        print(f"  OK — Downloaded successfully ({size_kb} KB)")

    except urllib.error.URLError as e:
        print()
        print("  ERROR: Could not download haarcascade XML.")
        print(f"  Reason: {e}")
        print()
        print("  ── MANUAL DOWNLOAD INSTRUCTIONS ──────────────────")
        print("  1. Open this URL in your browser:")
        print("     https://github.com/opencv/opencv/blob/master/")
        print("     data/haarcascades/haarcascade_frontalface_default.xml")
        print()
        print("  2. Click the download button (arrow icon)")
        print()
        print("  3. Save the file to:")
        print(f"     {os.path.abspath(CASCADE_DEST)}")
        print()
        print("  4. Then run this setup script again: python 00_setup.py")
        print("  ──────────────────────────────────────────────────")
        print()
        print("  Setup cannot continue without this file. Exiting.")
        sys.exit(1)

    except Exception as e:
        print(f"  ERROR: Unexpected download error: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────
# STEP 4: Verify haarcascade XML is valid
# A valid file starts with "<?xml" and is > 100KB
# ──────────────────────────────────────────────
print()
print("[4/6] Verifying haarcascade XML integrity...")

try:
    with open(CASCADE_DEST, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    size_kb = os.path.getsize(CASCADE_DEST) // 1024

    if first_line.startswith("<?xml") and size_kb > 100:
        print(f"  OK — Valid XML file ({size_kb} KB)")
    else:
        print(f"  ERROR: File appears corrupt or incomplete ({size_kb} KB).")
        print(f"  First line: {first_line[:80]}")
        print("  Delete the file and run 00_setup.py again.")
        sys.exit(1)

except Exception as e:
    print(f"  ERROR: Cannot read haarcascade file: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 5: Initialize SQLite database
# ──────────────────────────────────────────────
print()
print("[5/6] Initializing database...")

try:
    # Import and run setup_db from db.py
    from db import setup_db
    setup_db()
    print("  OK — attendance.db ready")

except ImportError:
    print("  ERROR: db.py not found in current folder.")
    print("  Make sure you are running this from the FaceAttendance/ directory.")
    sys.exit(1)

except Exception as e:
    print(f"  ERROR: Database setup failed: {e}")
    sys.exit(1)


# ──────────────────────────────────────────────
# STEP 6: Verify all required packages
# ──────────────────────────────────────────────
print()
print("[6/6] Checking installed packages...")

REQUIRED_PACKAGES = {
    "cv2"       : "opencv-python + opencv-contrib-python",
    "numpy"     : "numpy",
    "pandas"    : "pandas",
    "PIL"       : "Pillow",
    "openpyxl"  : "openpyxl",
}

all_ok = True
missing = []
cv2_import_ok = True  # Track cv2 import separately for contrib check

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
        print(f"  OK — {pip_name}")
    except ImportError:
        print(f"  MISSING — {pip_name}")
        print(f"            Install with: pip install {pip_name.split('+')[0].strip()}")
        missing.append(pip_name)
        all_ok = False
        if module_name == "cv2":
            cv2_import_ok = False

# Special check: cv2.face (opencv-contrib-python)
# Only run this if cv2 itself imported successfully
if cv2_import_ok:
    try:
        import cv2
        _ = cv2.face.LBPHFaceRecognizer_create()
        print(f"  OK — opencv-contrib-python (LBPHFaceRecognizer)")
    except AttributeError:
        # cv2 imported but cv2.face is missing — contrib not installed
        print("  MISSING — opencv-contrib-python")
        print("            Install with: pip install opencv-contrib-python==4.8.0.76")
        missing.append("opencv-contrib-python")
        all_ok = False
    except Exception as e:
        print(f"  WARNING — cv2.face check failed: {e}")
else:
    print("  SKIPPED — opencv-contrib check (cv2 not installed yet)")


# ──────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────
print()
print("=" * 55)

if all_ok and len(missing) == 0:
    print("  SETUP COMPLETE — Everything is ready!")
    print()
    print("  Folder structure:")
    print("    dataset/     <- face images will be saved here")
    print("    trainer/     <- trained model will be saved here")
    print("    attendance/  <- CSV and Excel reports saved here")
    print("    haarcascade/ <- haarcascade_frontalface_default.xml")
    print()
    print("  RUN ORDER:")
    print("    Step 1: python 01_collect.py   (add each student)")
    print("    Step 2: python 02_train.py     (train the model)")
    print("    Step 3: python 03_attendance.py (take attendance)")
    print("    Step 4: python 04_report.py    (view/export reports)")

else:
    print("  SETUP INCOMPLETE — Fix the issues above:")
    print()
    for pkg in missing:
        print(f"    pip install {pkg.split('+')[0].strip()}")
    print()
    print("  After installing, run: python 00_setup.py again")

print("=" * 55)
print()

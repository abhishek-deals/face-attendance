# Face Recognition Attendance System
### Lightweight | LBPH | SQLite | OpenCV Only
### Optimized for AMD Athlon Silver 3050U · 4GB RAM · No GPU

---

## Quick Start

```
STEP 1 — Install packages (one by one):
   pip install opencv-python
   pip install opencv-contrib-python
   pip install numpy
   pip install pandas
   pip install Pillow
   pip install openpyxl

STEP 2 — Download haarcascade XML:
   URL: https://github.com/opencv/opencv/tree/master/data/haarcascades
   File: haarcascade_frontalface_default.xml
   Place at: FaceAttendance/haarcascade/haarcascade_frontalface_default.xml

STEP 3 — Add each student (run once per student):
   python 01_collect.py

STEP 4 — Train model (after adding ALL students):
   python 02_train.py

STEP 5 — Run attendance (daily):
   python 03_attendance.py

STEP 6 — View/Export reports:
   python 04_report.py
```

---

## Folder Structure

```
FaceAttendance/
├── dataset/          ← face images (auto-created)
│   └── 101_Alice/    ← 50 x 100x100 JPG images
├── trainer/          ← trained model (auto-created)
│   ├── model.yml     ← LBPH model
│   └── names.txt     ← ID-to-name mapping
├── attendance/       ← Excel exports (auto-created)
├── haarcascade/      ← PUT XML HERE
│   └── haarcascade_frontalface_default.xml
├── db.py             ← database functions
├── 01_collect.py     ← collect face images
├── 02_train.py       ← train LBPH model
├── 03_attendance.py  ← real-time attendance
├── 04_report.py      ← view and export reports
├── attendance.db     ← SQLite database (auto-created)
└── requirements.txt
```

---

## Performance Settings (for 4GB RAM / AMD Athlon)

| Setting             | Value   | Why                            |
|---------------------|---------|--------------------------------|
| Camera resolution   | 320×240 | 4× less data than 640×480     |
| Camera FPS          | 15      | Reduces heat vs 30 FPS        |
| Frame skip          | 1 of 3  | 66% less CPU processing        |
| Grayscale only      | Yes     | 1/3 the memory of color       |
| Face save size      | 100×100 | Tiny, enough for LBPH          |
| Images per student  | 50      | Sufficient accuracy, less disk |
| Algorithm           | LBPH    | Lightest ML, no GPU needed     |

---

## Tuning for Hot/Slow Laptop

```python
# In 03_attendance.py — increase skip to cool down:
SKIP_FRAMES = 5       # was 3

# Reduce FPS if camera lags:
cam.set(cv2.CAP_PROP_FPS, 10)  # was 15

# In 01_collect.py — reduce images if storage is tight:
TARGET = 30           # was 50

# If recognition misses faces:
CONFIDENCE_LIMIT = 80  # was 70 (more lenient)
```

---

## Before Running — Free Up RAM

1. Close Chrome / Edge completely
2. Close music / video apps
3. Open Task Manager → ensure RAM < 80%
4. Disable Windows Search indexing temporarily

---

## Storage Estimate

| Students | Images       | Dataset Size | Model Size |
|----------|-------------|--------------|------------|
| 10       | 500 total   | ~5 MB        | ~0.5 MB    |
| 20       | 1000 total  | ~10 MB       | ~1 MB      |
| 50       | 2500 total  | ~25 MB       | ~2 MB      |

---

## What is NOT Used (and why)

| Library         | Reason Excluded                          |
|-----------------|------------------------------------------|
| TensorFlow      | 2GB+ RAM, needs GPU                      |
| Keras           | Too heavy for 4GB RAM                    |
| face_recognition| Requires dlib — hard to compile          |
| dlib            | C++ compile required, heavy RAM          |
| mediapipe       | Google ML — too heavy without GPU        |
| Tkinter GUI     | Causes lag on integrated graphics        |
| PyTorch          | 1GB+ install, needs GPU for speed        |

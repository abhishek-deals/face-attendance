import os
import sys
import json
import base64
import importlib.util

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

# CASCADE is a static file bundled in the repo
os.environ["CASCADE_PATH"] = os.path.join(project_root, "haarcascade", "haarcascade_frontalface_default.xml")

# /tmp is used only as scratch space for OpenCV model read/write (not persistence)
os.makedirs("/tmp/scratch", exist_ok=True)

from flask import Flask, request, jsonify, Response, send_file
import io

# ── import the dashboard for HTML page generation
dashboard_path = os.path.join(project_root, "05_dashboard.py")
spec = importlib.util.spec_from_file_location("dashboard", dashboard_path)
dashboard = importlib.util.module_from_spec(spec)
sys.modules["dashboard"] = dashboard
spec.loader.exec_module(dashboard)

# ── import the persistent database module
import pdb as pstore

# Initialise DB schema on cold start (idempotent)
try:
    pstore.setup_db()
except Exception as e:
    print("[WARN] DB setup error:", e)

app = Flask(__name__)

# ─────────────────────────────────────────
# HTML Pages (delegated to dashboard.py)
# ─────────────────────────────────────────

@app.route('/')
def route_today():
    return dashboard.page_today()

@app.route('/all')
def route_all():
    return dashboard.page_all()

@app.route('/students')
def route_students():
    return dashboard.page_students()

@app.route('/register')
def route_register():
    return dashboard.page_register()

@app.route('/live')
def route_live():
    return dashboard.page_live()

@app.route('/train')
def route_train():
    return getattr(dashboard, "page_train", lambda: "Not Implemented")()

@app.route('/student_details')
def route_student_details():
    sid = request.args.get('id', '')
    return dashboard.page_student_details(sid)

@app.route('/logo.png')
def route_logo():
    logo_path = os.path.join(project_root, "logo.png")
    if os.path.exists(logo_path):
        return send_file(logo_path, mimetype='image/png')
    return "Not found", 404

@app.route('/photo')
def route_photo():
    """Serve student profile photo from persistent DB."""
    sid = request.args.get('id', '')
    if not sid:
        return "Not found", 404
    try:
        photo_bytes = pstore.get_first_face_photo(sid)
        if photo_bytes:
            return Response(photo_bytes, mimetype='image/jpeg')
    except Exception:
        pass
    return "Not found", 404

# ─────────────────────────────────────────
# API: Capture Frame (registration)
# ─────────────────────────────────────────

@app.route('/api/capture_frame', methods=['POST'])
def api_capture_frame():
    try:
        import cv2
        import numpy as np

        data = json.loads(request.get_data().decode("utf-8"))
        frame_b64     = data.get("frame", "")
        sid           = str(data.get("student_id", "")).strip()
        name          = str(data.get("name", "")).strip().replace(" ", "_")
        face_detected = data.get("face_detected", True)

        if not sid or not name or not frame_b64:
            return jsonify({"ok": False, "error": "Missing fields"})
        if not sid.isdigit():
            return jsonify({"ok": False, "error": "Invalid student ID"})
        if not face_detected:
            count = pstore.get_face_photo_count(sid)
            return jsonify({"ok": True, "face_found": False, "count": count})

        # Decode base64 image
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)

        # Decode to OpenCV image
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            count = pstore.get_face_photo_count(sid)
            return jsonify({"ok": True, "face_found": False, "count": count})

        # Convert to grayscale and equalise histogram for better matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # normalise brightness

        # Detect face with haarcascade
        cascade_path = os.environ.get("CASCADE_PATH", "")
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )

        if len(faces) == 0:
            count = pstore.get_face_photo_count(sid)
            return jsonify({"ok": True, "face_found": False, "count": count})

        # Use the largest detected face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (100, 100))

        # Encode resized face as JPEG bytes
        _, face_encoded = cv2.imencode('.jpg', face_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        face_bytes = face_encoded.tobytes()

        # Check how many photos already stored
        existing = pstore.get_face_photo_count(sid)
        if existing >= 3:
            pstore.add_student(sid, name.replace("_", " "))
            return jsonify({"ok": True, "face_found": True, "count": existing})

        # Save the face CROP (not the full frame) to persistent DB
        pstore.save_face_photo(sid, existing + 1, face_bytes)
        new_count = existing + 1

        if new_count >= 3:
            pstore.add_student(sid, name.replace("_", " "))

        return jsonify({"ok": True, "face_found": True, "count": new_count})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────
# API: Train Model
# ─────────────────────────────────────────

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Load all face crop photos from persistent DB
        photos = pstore.get_all_face_photos()
        if not photos:
            return jsonify({"ok": False, "error": "No training images found. Register students first."})

        # LBPH with parameters tuned for 100x100 face crops
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )

        face_samples = []
        ids = []
        id_name_map = {}

        for sid, sname, photo_bytes in photos:
            try:
                sid_int = int(sid)
                id_name_map[sid_int] = sname
                # Open as grayscale and resize to 100x100 for consistency
                pil_img = Image.open(io.BytesIO(photo_bytes)).convert("L").resize((100, 100))
                img_array = np.array(pil_img, dtype="uint8")
                # Apply histogram equalisation so lighting differences don't confuse the model
                img_array = cv2.equalizeHist(img_array)
                face_samples.append(img_array)
                ids.append(sid_int)
            except Exception:
                pass

        if not face_samples:
            return jsonify({"ok": False, "error": "No valid training images found."})

        recognizer.train(face_samples, np.array(ids))

        # Save model bytes to DB
        model_tmp = "/tmp/scratch/model.yml"
        recognizer.write(model_tmp)
        with open(model_tmp, "rb") as f:
            model_bytes = f.read()

        pstore.save_model(model_bytes, id_name_map)

        # Reset the in-memory recognizer cache so next recognition reloads the new model
        global _recognizer_cache, _names_cache
        _recognizer_cache = None
        _names_cache = None

        student_names = list(id_name_map.values())
        msg = f"Training complete! {len(student_names)} student(s): {', '.join(student_names)}"
        return jsonify({"ok": True, "students": student_names, "message": msg})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────
# API: Recognize Face (live attendance)
# ─────────────────────────────────────────

_recognizer_cache = None
_names_cache = None
_confirm_buffer = {}   # track consecutive matches per face label

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    global _recognizer_cache, _names_cache, _confirm_buffer
    try:
        import cv2
        import numpy as np

        # Load model from persistent DB if not cached
        if _recognizer_cache is None:
            model_bytes, names_dict = pstore.load_model()
            if model_bytes is None:
                return jsonify({"ok": False, "error": "Model not trained yet! Please train the model first."})
            model_tmp = "/tmp/scratch/model.yml"
            with open(model_tmp, "wb") as f:
                f.write(model_bytes)
            _recognizer_cache = cv2.face.LBPHFaceRecognizer_create()
            _recognizer_cache.read(model_tmp)
            _names_cache = {int(k): v for k, v in names_dict.items()}
            _confirm_buffer = {}

        data = json.loads(request.get_data().decode("utf-8"))
        frame_b64 = data.get("frame", "")
        if not frame_b64:
            return jsonify({"ok": False, "error": "No frame"})

        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"ok": True, "name": None})

        # Grayscale + histogram equalise (must match training preprocessing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        cascade_path = os.environ.get("CASCADE_PATH", "")
        detector = cv2.CascadeClassifier(cascade_path)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            _confirm_buffer = {}
            return jsonify({"ok": True, "name": None})

        # Largest face, centered check
        fh, fw = gray.shape
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_cx = x + w // 2
        face_cy = y + h // 2
        if abs(face_cx - fw // 2) > fw * 0.45 or abs(face_cy - fh // 2) > fh * 0.50:
            _confirm_buffer = {}
            return jsonify({"ok": True, "name": None})

        face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        label, conf = _recognizer_cache.predict(face_crop)

        # Tighter threshold: 80 works well for proper face crops with equalisation
        if conf < 80:
            name = _names_cache.get(label, "Unknown")
            if name != "Unknown":
                # Require 3 consecutive matching frames before marking attendance
                _confirm_buffer[label] = _confirm_buffer.get(label, 0) + 1
                # Reset other labels
                for k in list(_confirm_buffer.keys()):
                    if k != label:
                        _confirm_buffer[k] = 0

                if _confirm_buffer[label] >= 3:
                    pstore.mark_attendance(str(label), name)
                    return jsonify({"ok": True, "name": name, "conf": round(conf, 1), "confirmed": True})
                else:
                    # Not confirmed yet — show as scanning
                    return jsonify({"ok": True, "name": None, "pending": name,
                                    "frames": _confirm_buffer[label], "conf": round(conf, 1)})
            return jsonify({"ok": True, "name": name, "conf": round(conf, 1)})
        else:
            _confirm_buffer = {}
            return jsonify({"ok": True, "name": "Unknown", "conf": round(conf, 1)})

    except Exception as e:
        _recognizer_cache = None
        _confirm_buffer = {}
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────
# API: Delete Student
# ─────────────────────────────────────────

@app.route('/api/delete_student', methods=['POST'])
def api_delete_student():
    try:
        data = json.loads(request.get_data().decode("utf-8"))
        sid = str(data.get("student_id", "")).strip()
        pwd = data.get("password", "")

        if pwd != "vercel":
            return jsonify({"ok": False, "error": "Incorrect password!"})
        if not sid:
            return jsonify({"ok": False, "error": "Missing student ID"})

        pstore.delete_student(sid)
        global _recognizer_cache
        _recognizer_cache = None  # Force model reload after deletion
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────
# API: Status & Check Student
# ─────────────────────────────────────────

@app.route('/api/status')
def api_status():
    return jsonify({"ok": True, "running": False, "done": True, "message": "", "students": []})

@app.route('/api/check_student')
def api_check_student():
    sid = request.args.get('id', '')
    try:
        students = pstore.get_students_list()
        for s in students:
            if s['id'] == sid:
                return jsonify({"exists": True, "name": s['name']})
    except Exception:
        pass
    return jsonify({"exists": False})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

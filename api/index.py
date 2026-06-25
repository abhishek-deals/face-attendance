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

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        TARGET_PHOTOS = 8
        existing = pstore.get_face_photo_count(sid)
        if existing >= TARGET_PHOTOS:
            pstore.add_student(sid, name.replace("_", " "))
            return jsonify({"ok": True, "face_found": True, "count": existing, "target": TARGET_PHOTOS})

        # Save the face CROP (not the too large full frame) to persistent DB
        pstore.save_face_photo(sid, existing + 1, face_bytes)
        new_count = existing + 1

        if new_count >= TARGET_PHOTOS:
            pstore.add_student(sid, name.replace("_", " "))

        return jsonify({"ok": True, "face_found": True, "count": new_count, "target": TARGET_PHOTOS})

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

        # Inline augmentation helper
        # Each real photo generates ~11 variants so 8 photos
        # -> 88 training samples, matching 50+ manual captures.
        def _augment(img):
            variants = [img]
            h, w = img.shape
            ctr = (w // 2, h // 2)
            for f in [0.70, 0.85, 1.15, 1.30]:
                variants.append(
                    np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
                )
            for ang in [-10, -5, 5, 10]:
                M = cv2.getRotationMatrix2D(ctr, ang, 1.0)
                variants.append(
                    cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                )
            variants.append(cv2.flip(img, 1))
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            variants.append(
                np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            )
            return variants  # 11 variants

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
                # Apply CLAHE so lighting differences don't confuse the model
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)
                
                # Apply data augmentation
                for variant in _augment(img_array):
                    face_samples.append(variant)
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
_confirm_buffer = {}   # {client_ip: {"label": int, "count": int, "conf_sum": float}}
_CONFIRM_FRAMES = 5    # Must see same person this many frames in a row
_CONFIDENCE_THRESHOLD = 60  # LBPH: lower = better match. Reject anything >= this.
                             # 60 = good balance: avoids wrong names, works in varied lighting.

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    global _recognizer_cache, _names_cache, _confirm_buffer
    try:
        import cv2
        import numpy as np

        # Get client IP for buffering
        client_key = request.headers.get('x-forwarded-for', request.remote_addr)
        if not client_key:
            client_key = "default"

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
            # Ensure names are in Title Case (handles ALL_CAPS stored names)
            _names_cache = {int(k): v.replace("_", " ").title() for k, v in names_dict.items()}
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
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        # Grayscale (must match training preprocessing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade_path = os.environ.get("CASCADE_PATH", "")
        detector = cv2.CascadeClassifier(cascade_path)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        # Largest face, centered check
        fh, fw = gray.shape
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_cx = x + w // 2
        face_cy = y + h // 2
        if abs(face_cx - fw // 2) > fw * 0.40 or abs(face_cy - fh // 2) > fh * 0.45:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        # Match api_train: apply CLAHE on the resized crop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_crop = clahe.apply(face_crop)
        label, conf = _recognizer_cache.predict(face_crop)

        # Confidence gate: reject poor matches
        if conf >= _CONFIDENCE_THRESHOLD:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": "Unknown", "conf": round(conf, 1)})

        name = _names_cache.get(label, "Unknown")

        # Multi-frame confirmation buffer
        buf = _confirm_buffer.get(client_key, {"label": -1, "count": 0, "conf_sum": 0.0})

        # If already marked this session (cooldown sentinel count=999), return already_marked
        if buf.get("marked") and buf["label"] == label:
            return jsonify({"ok": True, "name": name, "conf": round(conf, 1), "already_marked": True})

        if buf["label"] == label and not buf.get("marked"):
            buf["count"] += 1
            buf["conf_sum"] += conf
        else:
            buf = {"label": label, "count": 1, "conf_sum": conf}

        _confirm_buffer[client_key] = buf

        avg_conf = round(buf["conf_sum"] / buf["count"], 1)

        if buf["count"] < _CONFIRM_FRAMES:
            return jsonify({
                "ok": True,
                "name": None,
                "pending": name,
                "frames": buf["count"],
                "needed": _CONFIRM_FRAMES,
                "conf": avg_conf
            })

        # Confirmed! Mark attendance
        _confirm_buffer.pop(client_key, None)
        if name != "Unknown":
            pstore.mark_attendance(str(label), name)
            # Set cooldown so same person isn't re-announced in same session
            _confirm_buffer[client_key] = {"label": label, "count": 999, "conf_sum": 0.0, "marked": True}
        return jsonify({"ok": True, "name": name, "conf": avg_conf, "confirmed": True})

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

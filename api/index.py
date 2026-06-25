import os
import sys
import json
import base64
import importlib.util

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# CASCADE bundled in repo — set before importing anything
os.environ["CASCADE_PATH"] = os.path.join(
    project_root, "haarcascade", "haarcascade_frontalface_default.xml"
)

# Scratch dir for OpenCV model file (write-only, not persistent)
os.makedirs("/tmp/scratch", exist_ok=True)

from flask import Flask, request, jsonify, Response, send_file
import io

# ── Load dashboard module for HTML page rendering
dashboard_path = os.path.join(project_root, "05_dashboard.py")
spec = importlib.util.spec_from_file_location("dashboard", dashboard_path)
dashboard = importlib.util.module_from_spec(spec)
sys.modules["dashboard"] = dashboard
try:
    spec.loader.exec_module(dashboard)
except Exception as _e:
    print("[WARN] dashboard load error:", _e)

# ── Load persistent store (pdb.py handles PostgreSQL + SQLite fallback)
import pdb as pstore

# Initialise DB schema on cold start (safe to call multiple times)
try:
    pstore.setup_db()
    print("[OK] DB setup complete. Backend:", "PostgreSQL" if pstore._use_postgres() else "SQLite(/tmp)")
except Exception as _e:
    print("[WARN] DB setup error:", _e)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HTML Pages  (all delegated to dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def route_today():
    try:
        return dashboard.page_today()
    except Exception as e:
        return _err_page("Today", str(e))

@app.route('/all')
def route_all():
    try:
        return dashboard.page_all()
    except Exception as e:
        return _err_page("All Records", str(e))

@app.route('/students')
def route_students():
    try:
        return dashboard.page_students()
    except Exception as e:
        return _err_page("Students", str(e))

@app.route('/register')
def route_register():
    try:
        return dashboard.page_register()
    except Exception as e:
        return _err_page("Register", str(e))

@app.route('/live')
def route_live():
    try:
        return dashboard.page_live()
    except Exception as e:
        return _err_page("Live Attendance", str(e))

@app.route('/train')
def route_train():
    try:
        fn = getattr(dashboard, "page_train", None)
        return fn() if fn else "Not Implemented"
    except Exception as e:
        return _err_page("Train Model", str(e))

@app.route('/student_details')
def route_student_details():
    sid = request.args.get('id', '')
    try:
        return dashboard.page_student_details(sid)
    except Exception as e:
        return _err_page("Student Details", str(e))

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

# ─────────────────────────────────────────────────────────────────────────────
# Helper: simple error page
# ─────────────────────────────────────────────────────────────────────────────
def _err_page(title, msg):
    return f"""<!DOCTYPE html><html><head><title>{title} Error</title>
<style>body{{background:#09090b;color:#f8fafc;font-family:sans-serif;padding:40px}}
.err{{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.3);
border-radius:12px;padding:24px;color:#fca5a5}}</style></head>
<body><h2 style="color:#a78bfa">{title}</h2>
<div class="err"><strong>Error:</strong> {msg}</div>
<a href="/" style="color:#8b5cf6;margin-top:20px;display:block">← Home</a>
</body></html>""", 500

# ─────────────────────────────────────────────────────────────────────────────
# API: Capture Frame  (student registration)
# ─────────────────────────────────────────────────────────────────────────────

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
            return jsonify({"ok": False, "error": "Invalid student ID (must be numeric)"})

        current_count = pstore.get_face_photo_count(sid)

        if not face_detected:
            return jsonify({"ok": True, "face_found": False, "count": current_count})

        # Strip data-URL prefix
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)

        # Decode to OpenCV image
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"ok": True, "face_found": False, "count": current_count})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection with Haar cascade
        cascade_path = os.environ.get("CASCADE_PATH", "")
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) == 0:
            return jsonify({"ok": True, "face_found": False, "count": current_count})

        # Largest detected face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (100, 100))

        # CLAHE normalization (matches training pipeline)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_resized = clahe.apply(face_resized)

        # Encode as JPEG bytes
        _, face_encoded = cv2.imencode('.jpg', face_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        face_bytes = face_encoded.tobytes()

        TARGET_PHOTOS = 8
        existing = pstore.get_face_photo_count(sid)

        if existing >= TARGET_PHOTOS:
            # Already have enough photos — ensure student is registered
            pstore.add_student(sid, name.replace("_", " "))
            return jsonify({"ok": True, "face_found": True, "count": existing, "target": TARGET_PHOTOS, "done": True})

        # Save face photo to DB
        pstore.save_face_photo(sid, existing + 1, face_bytes)
        new_count = existing + 1

        if new_count >= TARGET_PHOTOS:
            pstore.add_student(sid, name.replace("_", " "))

        return jsonify({
            "ok": True,
            "face_found": True,
            "count": new_count,
            "target": TARGET_PHOTOS,
            "done": new_count >= TARGET_PHOTOS
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# API: Train Model
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Load all face crop photos from DB
        photos = pstore.get_all_face_photos()
        if not photos:
            return jsonify({"ok": False, "error": "No training photos found. Register students first via the Register page."})

        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )

        def _augment(img):
            """Each photo → ~11 variants. 8 photos → 88 training samples."""
            variants = [img]
            h, w = img.shape
            ctr = (w // 2, h // 2)
            for f in [0.70, 0.85, 1.15, 1.30]:
                variants.append(np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8))
            for ang in [-10, -5, 5, 10]:
                M = cv2.getRotationMatrix2D(ctr, ang, 1.0)
                variants.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
            variants.append(cv2.flip(img, 1))
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            variants.append(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
            return variants  # 11 variants total

        face_samples = []
        ids = []
        id_name_map = {}

        for sid, sname, photo_bytes in photos:
            try:
                sid_int = int(sid)
                id_name_map[sid_int] = sname
                pil_img = Image.open(io.BytesIO(photo_bytes)).convert("L").resize((100, 100))
                img_array = np.array(pil_img, dtype="uint8")
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)
                for variant in _augment(img_array):
                    face_samples.append(variant)
                    ids.append(sid_int)
            except Exception as _e:
                print(f"[WARN] Skipping photo for sid={sid}: {_e}")

        if not face_samples:
            return jsonify({"ok": False, "error": "No valid face images could be processed. Try re-registering students."})

        recognizer.train(face_samples, np.array(ids))
        del face_samples, ids  # Free RAM

        # Serialize model to tmp file then store bytes in DB
        model_tmp = "/tmp/scratch/model.yml"
        recognizer.write(model_tmp)
        with open(model_tmp, "rb") as f:
            model_bytes = f.read()

        # Store model in DB along with name mapping
        pstore.save_model(model_bytes, id_name_map)

        # Invalidate in-memory recognizer cache
        global _recognizer_cache, _names_cache
        _recognizer_cache = None
        _names_cache = None

        student_names = [str(v).replace("_", " ").title() for v in id_name_map.values()]
        return jsonify({
            "ok": True,
            "students": student_names,
            "message": f"Training complete! {len(student_names)} student(s): {', '.join(student_names)}"
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# API: Recognize Face  (live attendance)
# ─────────────────────────────────────────────────────────────────────────────

_recognizer_cache = None
_names_cache      = None
_confirm_buffer   = {}   # {client_key: {"label": int, "count": int, "conf_sum": float, "marked": bool}}
_CONFIRM_FRAMES         = 5   # consecutive frames required before marking
_CONFIDENCE_THRESHOLD   = 62  # LBPH distance — lower = better match; reject >= this

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    global _recognizer_cache, _names_cache, _confirm_buffer
    try:
        import cv2
        import numpy as np

        client_key = request.headers.get('x-forwarded-for', request.remote_addr) or "default"

        # Load recognizer from DB if not already in memory
        if _recognizer_cache is None:
            model_bytes, names_dict = pstore.load_model()
            if model_bytes is None:
                return jsonify({"ok": False, "error": "Model not trained yet! Please go to Train Model and click Start Training first."})
            model_tmp = "/tmp/scratch/model.yml"
            with open(model_tmp, "wb") as f:
                f.write(model_bytes)
            _recognizer_cache = cv2.face.LBPHFaceRecognizer_create()
            _recognizer_cache.read(model_tmp)
            _names_cache = {int(k): str(v).replace("_", " ").title() for k, v in names_dict.items()}
            _confirm_buffer = {}
            print(f"[OK] Model loaded. Students: {list(_names_cache.values())}")

        data = json.loads(request.get_data().decode("utf-8"))
        frame_b64 = data.get("frame", "")
        if not frame_b64:
            return jsonify({"ok": False, "error": "No frame provided"})

        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(frame_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade_path = os.environ.get("CASCADE_PATH", "")
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=7,
            minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        # Pick largest face and check it's roughly centered
        fh, fw = gray.shape
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_cx, face_cy = x + w // 2, y + h // 2
        if abs(face_cx - fw // 2) > fw * 0.42 or abs(face_cy - fh // 2) > fh * 0.48:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        # Preprocess: CLAHE + resize (must match training)
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_crop = clahe.apply(face_crop)

        label, conf = _recognizer_cache.predict(face_crop)

        # Confidence gate
        if conf >= _CONFIDENCE_THRESHOLD:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": "Unknown", "conf": round(conf, 1)})

        name = _names_cache.get(label, "Unknown")
        buf  = _confirm_buffer.get(client_key, {"label": -1, "count": 0, "conf_sum": 0.0})

        # Cooldown: same person already marked this session
        if buf.get("marked") and buf["label"] == label:
            return jsonify({"ok": True, "name": name, "conf": round(conf, 1), "already_marked": True})

        if buf["label"] == label and not buf.get("marked"):
            buf["count"]    += 1
            buf["conf_sum"] += conf
        else:
            buf = {"label": label, "count": 1, "conf_sum": conf}

        _confirm_buffer[client_key] = buf
        avg_conf = round(buf["conf_sum"] / buf["count"], 1)

        if buf["count"] < _CONFIRM_FRAMES:
            return jsonify({
                "ok": True, "name": None,
                "pending": name, "frames": buf["count"],
                "needed": _CONFIRM_FRAMES, "conf": avg_conf
            })

        # ✅ Confirmed — mark attendance
        _confirm_buffer.pop(client_key, None)
        if name != "Unknown":
            pstore.mark_attendance(str(label), name)
            _confirm_buffer[client_key] = {"label": label, "count": 999, "conf_sum": 0.0, "marked": True}

        return jsonify({"ok": True, "name": name, "conf": avg_conf, "confirmed": True})

    except Exception as e:
        _recognizer_cache = None
        _confirm_buffer   = {}
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# API: Delete Student
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/delete_student', methods=['POST'])
def api_delete_student():
    try:
        data = json.loads(request.get_data().decode("utf-8"))
        sid  = str(data.get("student_id", "")).strip()
        pwd  = data.get("password", "")

        if pwd != "vercel":
            return jsonify({"ok": False, "error": "Incorrect password! (hint: vercel)"})
        if not sid:
            return jsonify({"ok": False, "error": "Missing student ID"})

        pstore.delete_student(sid)
        global _recognizer_cache
        _recognizer_cache = None  # Force model reload after deletion
        return jsonify({"ok": True, "message": f"Student {sid} deleted successfully."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# API: Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/status')
def api_status():
    """Used by training page to poll status."""
    return jsonify({"ok": True, "running": False, "done": True, "message": "", "students": []})

@app.route('/api/check_student')
def api_check_student():
    """Check if a student ID already exists before registration."""
    sid = request.args.get('id', '')
    try:
        students = pstore.get_students_list()
        for s in students:
            if str(s['id']) == str(sid):
                return jsonify({"exists": True, "name": s['name']})
    except Exception:
        pass
    return jsonify({"exists": False})

@app.route('/api/db_info')
def api_db_info():
    """Debug: show which database backend is active."""
    try:
        backend = "PostgreSQL" if pstore._use_postgres() else f"SQLite ({pstore._SQLITE_PATH})"
        students = pstore.get_students_list()
        model_bytes, names = pstore.load_model()
        return jsonify({
            "ok": True,
            "backend": backend,
            "student_count": len(students),
            "model_trained": model_bytes is not None,
            "student_names": [s['name'] for s in students]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

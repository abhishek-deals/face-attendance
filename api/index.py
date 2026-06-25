import os
import sys
import json
import base64
import importlib.util
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Paths for Deep Learning ONNX Models (YuNet & SFace)
YUNET_PATH = os.path.join(project_root, "haarcascade", "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(project_root, "haarcascade", "face_recognition_sface_2021dec.onnx")

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

# Initialise DB schema on cold start
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
def route_today(): return dashboard.page_today()

@app.route('/all')
def route_all(): return dashboard.page_all()

@app.route('/students')
def route_students(): return dashboard.page_students()

@app.route('/register')
def route_register(): return dashboard.page_register()

@app.route('/live')
def route_live(): return dashboard.page_live()

@app.route('/train')
def route_train(): return getattr(dashboard, "page_train", lambda: "Not Implemented")()

@app.route('/student_details')
def route_student_details():
    sid = request.args.get('id', '')
    return dashboard.page_student_details(sid)

@app.route('/logo.png')
def route_logo():
    logo_path = os.path.join(project_root, "logo.png")
    if os.path.exists(logo_path): return send_file(logo_path, mimetype='image/png')
    return "Not found", 404

@app.route('/photo')
def route_photo():
    sid = request.args.get('id', '')
    if not sid: return "Not found", 404
    try:
        photo_bytes = pstore.get_first_face_photo(sid)
        if photo_bytes: return Response(photo_bytes, mimetype='image/jpeg')
    except Exception: pass
    return "Not found", 404

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
        
        current_count = pstore.get_face_photo_count(sid)
        if not face_detected:
            return jsonify({"ok": True, "face_found": False, "count": current_count})

        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"ok": True, "face_found": False, "count": current_count})

        # Deep Learning Face Detection (YuNet)
        h, w, _ = frame.shape
        detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (w, h), 0.8, 0.3, 5000)
        faces = detector.detect(frame)
        
        if faces[1] is None:
            return jsonify({"ok": True, "face_found": False, "count": current_count})

        # Get largest face
        face = sorted(faces[1], key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # Deep Learning Face Alignment (SFace)
        recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        aligned_face = recognizer.alignCrop(frame, face) # Outputs 112x112 perfect face crop

        # Save the 112x112 aligned face to DB
        _, face_encoded = cv2.imencode('.jpg', aligned_face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        TARGET_PHOTOS = 8
        existing = pstore.get_face_photo_count(sid)

        if existing >= TARGET_PHOTOS:
            pstore.add_student(sid, name.replace("_", " "))
            return jsonify({"ok": True, "face_found": True, "count": existing, "target": TARGET_PHOTOS, "done": True})

        pstore.save_face_photo(sid, existing + 1, face_encoded.tobytes())
        new_count = existing + 1

        if new_count >= TARGET_PHOTOS:
            pstore.add_student(sid, name.replace("_", " "))

        return jsonify({"ok": True, "face_found": True, "count": new_count, "target": TARGET_PHOTOS, "done": new_count >= TARGET_PHOTOS})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# API: Train Model (Extract SFace Embeddings)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        import cv2
        import numpy as np

        photos = pstore.get_all_face_photos()
        if not photos:
            return jsonify({"ok": False, "error": "No training photos found. Register students first."})

        recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        embeddings_dict = {}

        for sid, sname, photo_bytes in photos:
            try:
                sid_int = int(sid)
                nparr = np.frombuffer(photo_bytes, np.uint8)
                aligned_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Ensure correct size for SFace (112x112)
                if aligned_face.shape[:2] != (112, 112):
                    aligned_face = cv2.resize(aligned_face, (112, 112))
                
                # Extract 128-dimensional deep learning feature
                feat = recognizer.feature(aligned_face) # shape: (1, 128)
                
                if sid_int not in embeddings_dict:
                    embeddings_dict[sid_int] = {"name": sname, "features": []}
                embeddings_dict[sid_int]["features"].append(feat[0]) # store 1D array
                
            except Exception as _e:
                print(f"[WARN] Failed to extract feature for {sid}: {_e}")

        if not embeddings_dict:
            return jsonify({"ok": False, "error": "No valid face embeddings could be extracted."})

        # Serialize dictionary of embeddings
        model_bytes = pickle.dumps(embeddings_dict)
        
        # Save to DB (pstore.save_model signature: save_model(model_bytes, id_name_map))
        # Since we use embeddings_dict, we pass a dummy id_name_map
        id_name_map = {sid: data["name"] for sid, data in embeddings_dict.items()}
        pstore.save_model(model_bytes, id_name_map)

        # Invalidate cache
        global _embeddings_cache
        _embeddings_cache = None

        student_names = [str(data["name"]).replace("_", " ").title() for data in embeddings_dict.values()]
        return jsonify({
            "ok": True,
            "students": student_names,
            "message": f"Deep Learning Training complete! {len(student_names)} student(s) mapped."
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# API: Recognize Face  (live attendance)
# ─────────────────────────────────────────────────────────────────────────────

_embeddings_cache = None
_confirm_buffer   = {}   
_CONFIRM_FRAMES   = 4   # consecutive frames required
_SFACE_THRESHOLD  = 0.363  # Official Cosine distance threshold for SFace

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    global _embeddings_cache, _confirm_buffer
    try:
        import cv2
        import numpy as np

        client_key = request.headers.get('x-forwarded-for', request.remote_addr) or "default"

        # Load embeddings from DB
        if _embeddings_cache is None:
            model_bytes, _ = pstore.load_model()
            if model_bytes is None:
                return jsonify({"ok": False, "error": "Model not trained yet! Click Start Training first."})
            _embeddings_cache = pickle.loads(model_bytes)
            _confirm_buffer = {}

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

        h, w, _ = frame.shape
        detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (w, h), 0.8, 0.3, 5000)
        faces = detector.detect(frame)

        if faces[1] is None:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        # Pick largest face
        face = sorted(faces[1], key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # Check if reasonably centered
        face_cx, face_cy = face[0] + face[2] // 2, face[1] + face[3] // 2
        if abs(face_cx - w // 2) > w * 0.42 or abs(face_cy - h // 2) > h * 0.48:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": None})

        # Deep Learning Alignment & Feature Extraction
        recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        aligned_face = recognizer.alignCrop(frame, face)
        live_feat = recognizer.feature(aligned_face)[0] # 1D array of 128 elements

        # Find best match using Cosine Distance
        best_match = "Unknown"
        best_label = -1
        best_score = 1.0 # Cosine distance (lower is better, max is 2.0)

        live_norm = np.linalg.norm(live_feat)
        if live_norm > 0:
            for sid, stored_data in _embeddings_cache.items():
                for stored_feat in stored_data["features"]:
                    # Manual Cosine Distance: 1.0 - Cosine Similarity
                    cos_sim = np.dot(live_feat, stored_feat) / (live_norm * np.linalg.norm(stored_feat))
                    dist = 1.0 - cos_sim
                    
                    if dist < best_score:
                        best_score = dist
                        best_match = stored_data["name"]
                        best_label = sid

        # Convert distance to a human-readable 0-100% confidence score
        # 0.0 distance = 100%, 0.363 distance = ~60%
        conf_percent = max(0, int((1.0 - (best_score / _SFACE_THRESHOLD) * 0.4) * 100))

        if best_score > _SFACE_THRESHOLD:
            _confirm_buffer.pop(client_key, None)
            return jsonify({"ok": True, "name": "Unknown", "conf": conf_percent})

        name = str(best_match).replace("_", " ").title()
        label = best_label
        
        buf = _confirm_buffer.get(client_key, {"label": -1, "count": 0, "conf_sum": 0.0})

        if buf.get("marked") and buf["label"] == label:
            return jsonify({"ok": True, "name": name, "conf": conf_percent, "already_marked": True})

        if buf["label"] == label and not buf.get("marked"):
            buf["count"]    += 1
            buf["conf_sum"] += conf_percent
        else:
            buf = {"label": label, "count": 1, "conf_sum": conf_percent}

        _confirm_buffer[client_key] = buf
        avg_conf = int(buf["conf_sum"] / buf["count"])

        if buf["count"] < _CONFIRM_FRAMES:
            return jsonify({
                "ok": True, "name": None,
                "pending": name, "frames": buf["count"],
                "needed": _CONFIRM_FRAMES, "conf": avg_conf
            })

        # Confirmed
        _confirm_buffer.pop(client_key, None)
        if name != "Unknown":
            pstore.mark_attendance(str(label), name)
            _confirm_buffer[client_key] = {"label": label, "count": 999, "conf_sum": 0.0, "marked": True}

        return jsonify({"ok": True, "name": name, "conf": avg_conf, "confirmed": True})

    except Exception as e:
        _embeddings_cache = None
        _confirm_buffer   = {}
        return jsonify({"ok": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# API: Helpers
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/delete_student', methods=['POST'])
def api_delete_student():
    try:
        data = json.loads(request.get_data().decode("utf-8"))
        sid  = str(data.get("student_id", "")).strip()
        pwd  = data.get("password", "")

        if pwd != "vercel": return jsonify({"ok": False, "error": "Incorrect password! (hint: vercel)"})
        if not sid: return jsonify({"ok": False, "error": "Missing student ID"})

        pstore.delete_student(sid)
        global _embeddings_cache
        _embeddings_cache = None  # Force model reload
        return jsonify({"ok": True, "message": f"Student {sid} deleted successfully."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route('/api/status')
def api_status():
    return jsonify({"ok": True, "running": False, "done": True, "message": "", "students": []})

@app.route('/api/check_student')
def api_check_student():
    sid = request.args.get('id', '')
    try:
        for s in pstore.get_students_list():
            if str(s['id']) == str(sid): return jsonify({"exists": True, "name": s['name']})
    except Exception: pass
    return jsonify({"exists": False})

@app.route('/api/db_info')
def api_db_info():
    try:
        backend = "PostgreSQL" if pstore._use_postgres() else f"SQLite ({pstore._SQLITE_PATH})"
        students = pstore.get_students_list()
        model_bytes, names = pstore.load_model()
        return jsonify({"ok": True, "backend": backend, "student_count": len(students), "model_trained": model_bytes is not None})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

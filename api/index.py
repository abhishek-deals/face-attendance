import os
import sys
import importlib.util

# 1. Set environment variables to point to /tmp (since Vercel is read-only elsewhere)
os.environ["DB_PATH"] = "/tmp/attendance.db"
os.environ["ATTENDANCE_DIR"] = "/tmp/attendance"
os.environ["DATASET_PATH"] = "/tmp/dataset"
os.environ["TRAINER_PATH"] = "/tmp/trainer"
# We keep CASCADE_PATH looking in the project directory since it's read-only and static
project_root = os.path.dirname(os.path.dirname(__file__))
os.environ["CASCADE_PATH"] = os.path.join(project_root, "haarcascade", "haarcascade_frontalface_default.xml")

# Ensure /tmp directories exist
os.makedirs("/tmp/attendance", exist_ok=True)
os.makedirs("/tmp/dataset", exist_ok=True)
os.makedirs("/tmp/trainer", exist_ok=True)

# Also ensure db.py knows about DB_PATH if it reads it directly, though 
# we should make sure db.py uses the new os.environ.
# Let's patch db.py DB_PATH if needed or just copy the DB.
# For simplicity, if the project has db.py, let's copy it or import it to init.

from flask import Flask, request, jsonify, Response, send_file

# 2. Dynamically import 05_dashboard.py
sys.path.append(project_root)
dashboard_path = os.path.join(project_root, "05_dashboard.py")

spec = importlib.util.spec_from_file_location("dashboard", dashboard_path)
dashboard = importlib.util.module_from_spec(spec)
sys.modules["dashboard"] = dashboard
spec.loader.exec_module(dashboard)

# Setup DB if missing in /tmp
if not os.path.exists("/tmp/attendance.db"):
    try:
        import db
        # If db.py has a hardcoded path, we might need to force it
        if hasattr(db, 'DB_PATH'):
            db.DB_PATH = "/tmp/attendance.db"
        db.setup_db()
    except Exception as e:
        print("DB Setup error:", e)

# 3. Create Flask App
app = Flask(__name__)

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
    sid = request.args.get('id', '')
    if not sid:
        return "Not found", 404
        
    dataset = "/tmp/dataset"
    if not os.path.exists(dataset):
        return "Not found", 404
        
    try:
        sid_int = int(sid)
        for folder in os.listdir(dataset):
            parts = folder.split("_", 1)
            if len(parts) == 2 and int(parts[0]) == sid_int:
                fp = os.path.join(dataset, folder)
                imgs = [x for x in os.listdir(fp) if x.lower().endswith(".jpg")]
                if imgs:
                    return send_file(os.path.join(fp, imgs[0]), mimetype='image/jpeg')
    except Exception:
        pass
    return "Not found", 404

# API POST routes
@app.route('/api/capture_frame', methods=['POST'])
def api_capture_frame():
    return jsonify(dashboard.api_capture_frame(request.get_data()))

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    return jsonify(dashboard.api_recognize(request.get_data()))

@app.route('/api/delete_student', methods=['POST'])
def api_delete_student():
    return jsonify(dashboard.api_delete_student(request.get_data()))

@app.route('/api/train', methods=['POST'])
def api_train():
    # Because Vercel has a 10s timeout and we can't easily do background threads 
    # without returning, we run it synchronously. 
    # This might timeout on Vercel if there are too many students!
    result = dashboard.api_train()
    return jsonify(result)

@app.route('/api/status')
def api_status():
    return jsonify(dashboard.training_status)

@app.route('/api/check_student')
def api_check_student():
    sid = request.args.get('id', '')
    res = dashboard.db_query("SELECT name FROM students WHERE student_id=?", (sid,))
    if res:
        return jsonify({"exists": True, "name": res[0]['name']})
    return jsonify({"exists": False})

# Vercel requires the app variable to be exposed
if __name__ == '__main__':
    app.run(debug=True, port=5000)

# 05_dashboard.py — Web Dashboard + Student Registration
#
# ALL-IN-ONE: Reports dashboard + web-based student registration
# Uses ONLY Python built-in modules (no Flask, no extra pip install).
#
# PAGES:
#   http://localhost:5000/           → Today's attendance
#   http://localhost:5000/all        → All records
#   http://localhost:5000/students   → Registered students
#   http://localhost:5000/register   → Register new student (CAMERA + UPLOAD)
#   http://localhost:5000/train      → Train model after adding students
#
# HOW TO RUN:
#   python 05_dashboard.py
# ──────────────────────────────────────────────────────────────

import sqlite3
import os
import sys
import json
import base64
import threading
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

DB_PATH        = "attendance.db"
ATTENDANCE_DIR = "attendance"
DATASET_PATH   = "dataset"
TRAINER_PATH   = "trainer"
CASCADE_PATH   = os.path.join("haarcascade", "haarcascade_frontalface_default.xml")
PORT           = 5000

# Global training status
training_status = {"running": False, "done": False, "message": "", "students": []}


# ══════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════
def db_query(sql, params=()):
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()

def db_execute(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        return cur.rowcount
    except Exception:
        return 0
    finally:
        conn.close()

def get_today():
    return datetime.now().strftime("%Y-%m-%d")

def get_time():
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════
# SHARED HTML SHELL
# ══════════════════════════════════════════════
def html_page(title, body_content, active=""):
    nav_items = [
        ("/",         "Today"),
        ("/all",      "All Records"),
        ("/students", "Students"),
        ("/register", "Register Student"),
        ("/live",     "Live Attendance"),
        ("/train",    "Train Model"),
    ]
    nav_html = ""
    for href, label in nav_items:
        is_active = "active" if label == active else ""
        icon = {
            "Today": "📋",
            "All Records": "🗄️",
            "Students": "👥",
            "Register Student": "➕",
            "Live Attendance": "📹",
            "Train Model": "🧠",
        }.get(label, "")
        nav_html += f'<a href="{href}" class="{is_active}">{icon} {label}</a>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} | Face Attendance</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {{
  --bg: #09090b;
  --bg-glow: radial-gradient(circle at 50% -20%, #3b0764, #09090b 60%);
  --card: rgba(24, 24, 27, 0.7);
  --border: rgba(255, 255, 255, 0.08);
  --border-hover: rgba(255, 255, 255, 0.15);
  --accent: #8b5cf6;
  --accent-hover: #a78bfa;
  --accent2: #06b6d4;
  --text: #f8fafc;
  --muted: #94a3b8;
  --green: #10b981;
  --green-bg: rgba(16, 185, 129, 0.15);
  --red: #ef4444;
  --red-bg: rgba(239, 68, 68, 0.15);
  --yellow: #f59e0b;
  --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  --glow-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: var(--bg);
  background-image: var(--bg-glow);
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}}

/* TOPBAR */
.topbar {{
  background: rgba(9, 9, 11, 0.8);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 16px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}}
.topbar h1 {{
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #a78bfa, #22d3ee);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.topbar .badge {{
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  background: rgba(255, 255, 255, 0.05);
  padding: 6px 14px;
  border-radius: 30px;
  border: 1px solid var(--border);
}}

/* NAVIGATION */
nav {{
  background: rgba(24, 24, 27, 0.5);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  display: flex;
  gap: 4px;
  overflow-x: auto;
}}
nav a {{
  color: var(--muted);
  text-decoration: none;
  padding: 16px 20px;
  font-size: 14px;
  font-weight: 600;
  border-bottom: 2px solid transparent;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  white-space: nowrap;
}}
nav a:hover {{
  color: var(--text);
  background: rgba(255, 255, 255, 0.03);
}}
nav a.active {{
  color: var(--text);
  border-color: var(--accent);
  background: linear-gradient(to top, rgba(139, 92, 246, 0.15) 0%, transparent 100%);
  text-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
}}

/* CONTAINER & CARDS */
.container {{ max-width: 1200px; margin: 0 auto; padding: 40px 28px; }}

.card {{
  background: var(--card);
  backdrop-filter: blur(16px);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  margin-bottom: 24px;
  box-shadow: var(--glass-shadow);
  transition: border-color 0.3s;
}}
.card:hover {{
  border-color: var(--border-hover);
}}
.card-header {{
  padding: 20px 26px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(255,255,255,0.01);
}}
.card-header h2 {{ font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }}

/* STATS GRID */
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 32px; }}
.stat-card {{
  background: var(--card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  position: relative;
  overflow: hidden;
  box-shadow: var(--glass-shadow);
  transition: transform 0.3s, box-shadow 0.3s;
}}
.stat-card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}}
.stat-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; }}
.stat-card.green::before {{ background: var(--green); box-shadow: 0 0 10px var(--green); }}
.stat-card.purple::before {{ background: var(--accent); box-shadow: 0 0 10px var(--accent); }}
.stat-card.teal::before {{ background: var(--accent2); box-shadow: 0 0 10px var(--accent2); }}
.stat-card.yellow::before {{ background: var(--yellow); box-shadow: 0 0 10px var(--yellow); }}
.stat-label {{ font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}
.stat-value {{ font-size: 42px; font-weight: 800; margin: 8px 0 4px; letter-spacing: -1px; }}
.stat-value.green {{ color: var(--green); text-shadow: 0 0 20px rgba(16,185,129,0.3); }}
.stat-value.purple {{ color: var(--accent); text-shadow: 0 0 20px rgba(139,92,246,0.3); }}
.stat-value.teal {{ color: var(--accent2); text-shadow: 0 0 20px rgba(6,182,212,0.3); }}
.stat-value.yellow {{ color: var(--yellow); text-shadow: 0 0 20px rgba(245,158,11,0.3); }}
.stat-sub {{ font-size: 13px; font-weight: 500; color: var(--muted); }}

/* PILLS */
.pill {{ display: inline-flex; align-items: center; justify-content: center; padding: 4px 12px; border-radius: 30px; font-size: 12px; font-weight: 700; letter-spacing: 0.3px; }}
.pill.present {{ background: var(--green-bg); color: var(--green); border: 1px solid rgba(16,185,129,0.2); }}
.pill.id {{ background: rgba(139,92,246,0.15); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.2); }}
.pill.red {{ background: var(--red-bg); color: var(--red); border: 1px solid rgba(239,68,68,0.2); }}

/* TABLES */
table {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
th {{
  padding: 16px 22px; text-align: left; font-size: 12px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 1px; background: rgba(0,0,0,0.2);
  border-bottom: 1px solid var(--border);
}}
td {{ padding: 16px 22px; font-size: 14px; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.03); transition: background 0.2s; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: rgba(255,255,255,0.02); }}

/* BUTTONS */
.btn {{
  display: inline-flex; align-items: center; justify-content: center; padding: 12px 24px;
  border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; border: none;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-decoration: none; outline: none; gap: 8px;
}}
.btn-primary {{
  background: linear-gradient(135deg, var(--accent), #6d28d9);
  color: white; box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
}}
.btn-primary:hover {{
  transform: translateY(-2px); box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
}}
.btn-success {{
  background: linear-gradient(135deg, var(--green), #059669);
  color: white; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
}}
.btn-success:hover {{
  transform: translateY(-2px); box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
}}
.btn-outline {{
  background: rgba(255,255,255,0.02); border: 1px solid var(--border); color: var(--text);
}}
.btn-outline:hover {{
  background: rgba(255,255,255,0.05); border-color: var(--accent); color: var(--accent-hover);
  transform: translateY(-1px);
}}

/* FORMS */
.form-group {{ margin-bottom: 20px; }}
.form-group label {{
  display: block; font-size: 13px; font-weight: 600; color: var(--muted);
  margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;
}}
.form-group input {{
  width: 100%; background: rgba(0,0,0,0.3); border: 1px solid var(--border);
  color: var(--text); padding: 14px 18px; border-radius: 10px; font-size: 15px;
  transition: all 0.3s; outline: none; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}}
.form-group input:focus {{
  border-color: var(--accent); background: rgba(0,0,0,0.5); box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
}}

/* CAMERA & UPLOAD */
.reg-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
@media(max-width:800px) {{ .reg-grid {{ grid-template-columns: 1fr; }} }}
.cam-box {{
  background: #000; border-radius: 14px; overflow: hidden; position: relative;
  aspect-ratio: 4/3; border: 1px solid var(--border); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}}
.cam-box video {{ width: 100%; height: 100%; object-fit: cover; }}
.cam-box canvas {{ display: none; }}
.cam-overlay {{
  position: absolute; top: 14px; left: 14px; background: rgba(0,0,0,0.6);
  backdrop-filter: blur(4px); padding: 6px 14px; border-radius: 20px;
  font-size: 13px; font-weight: 700; color: white; border: 1px solid rgba(255,255,255,0.1);
}}
.upload-zone {{
  border: 2px dashed var(--border); border-radius: 14px; padding: 40px; text-align: center;
  cursor: pointer; transition: all 0.3s; background: rgba(255,255,255,0.01);
}}
.upload-zone:hover {{ border-color: var(--accent); background: rgba(139,92,246,0.05); transform: scale(1.02); }}

/* MISC */
.status-box {{ padding: 16px 20px; border-radius: 10px; font-size: 14px; font-weight: 500; margin-top: 16px; display: none; }}
.status-box.success {{ background: var(--green-bg); border: 1px solid rgba(16,185,129,0.3); color: var(--green); }}
.status-box.error {{ background: var(--red-bg); border: 1px solid rgba(239,68,68,0.3); color: #fca5a5; }}
.status-box.info {{ background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.3); color: #c4b5fd; }}

.empty {{ text-align: center; padding: 60px 20px; color: var(--muted); }}
.empty .icon {{ font-size: 56px; margin-bottom: 16px; opacity: 0.8; }}

.progress-bar {{ background: rgba(0,0,0,0.5); border-radius: 10px; height: 12px; overflow: hidden; margin: 16px 0; border: 1px solid var(--border); }}
.progress-fill {{ height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 10px; transition: width 0.4s ease-out; box-shadow: 0 0 10px rgba(139,92,246,0.5); }}

.tab-bar {{ display: flex; gap: 8px; margin-bottom: 24px; background: rgba(0,0,0,0.2); padding: 6px; border-radius: 12px; border: 1px solid var(--border); }}
.tab {{ flex: 1; padding: 12px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; border: none; background: transparent; color: var(--muted); transition: all 0.3s; }}
.tab.active {{ background: var(--card); color: var(--text); box-shadow: 0 4px 12px rgba(0,0,0,0.2); border: 1px solid var(--border-hover); }}

.preview-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 12px; margin-top: 16px; }}
.preview-grid img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px; border: 1px solid var(--border); transition: transform 0.2s; }}
.preview-grid img:hover {{ transform: scale(1.05); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}

footer {{ text-align: center; padding: 30px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--border); margin-top: 40px; font-weight: 500; }}
</style>
</head>
<body>
<div class="topbar">
  <h1 style="display:flex;align-items:center;gap:12px">
    <img src="/logo.png" style="width:34px;height:34px;border-radius:8px;box-shadow:0 2px 10px rgba(139,92,246,0.4)"> 
    Face Attendance System
  </h1>
  <div class="badge">{get_today()} &nbsp;|&nbsp; {get_time()}</div>
</div>
<nav>{nav_html}</nav>
<div class="container">
{body_content}
</div>
<footer>Face Recognition Attendance System &nbsp;|&nbsp; LBPH Algorithm &nbsp;|&nbsp; SQLite Database</footer>
</body>
</html>"""


# ══════════════════════════════════════════════
# PAGE: TODAY
# ══════════════════════════════════════════════
def page_today():
    today = get_today()
    records = db_query("SELECT student_id,name,date,time FROM attendance WHERE date=? ORDER BY time", (today,))
    total_s = db_query("SELECT COUNT(*) as c FROM students")[0]["c"] if db_query("SELECT COUNT(*) as c FROM students") else 0
    all_days = db_query("SELECT COUNT(DISTINCT date) as c FROM attendance")
    all_days = all_days[0]["c"] if all_days else 0

    stats = f"""<div class="stats-grid">
  <div class="stat-card green"><div class="stat-label">Present Today</div>
    <div class="stat-value green">{len(records)}</div><div class="stat-sub">of {total_s} registered</div></div>
  <div class="stat-card purple"><div class="stat-label">Total Students</div>
    <div class="stat-value purple">{total_s}</div><div class="stat-sub">enrolled</div></div>
  <div class="stat-card teal"><div class="stat-label">Total Sessions</div>
    <div class="stat-value teal">{all_days}</div><div class="stat-sub">days recorded</div></div>
  <div class="stat-card yellow"><div class="stat-label">Today</div>
    <div class="stat-value yellow" style="font-size:18px;margin-top:10px">{today}</div></div>
</div>"""

    if records:
        rows = "".join(f"""<tr><td style="color:var(--muted)">{i}</td>
          <td><img src="/photo?id={r['student_id']}" style="width:36px;height:36px;border-radius:50%;object-fit:cover" onerror="this.style.display='none'"></td>
          <td><span class="pill id">{r['student_id']}</span></td>
          <td><strong>{r['name']}</strong></td><td>{r['time']}</td>
          <td><span class="pill present">&#10003; Present</span></td></tr>"""
          for i,r in enumerate(records,1))
        table = f"""<div class="card">
  <div class="card-header"><h2>Today's Attendance &mdash; {today}</h2>
    <span class="pill present">{len(records)} Present</span></div>
  <table><thead><tr><th>#</th><th>Photo</th><th>ID</th><th>Name</th><th>Time</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody></table></div>
<p style="font-size:12px;color:var(--muted);text-align:right">Auto-refreshes every 30s</p>
<script>setTimeout(()=>location.reload(),30000);</script>"""
    else:
        table = """<div class="card"><div class="card-header"><h2>Today's Attendance</h2></div>
  <div class="empty"><div class="icon">&#128247;</div>
  <p>No attendance marked yet today.</p>
  <p style="margin-top:8px">Run <strong>python 03_attendance.py</strong> to start marking.</p></div></div>"""

    return html_page("Today", stats + table, "Today")


# ══════════════════════════════════════════════
# PAGE: ALL RECORDS
# ══════════════════════════════════════════════
def page_all():
    records = db_query("SELECT student_id,name,date,time FROM attendance ORDER BY date DESC,time DESC")
    if records:
        rows = "".join(f"""<tr><td style="color:var(--muted)">{i}</td>
          <td><span class="pill id">{r['student_id']}</span></td>
          <td><strong>{r['name']}</strong></td><td>{r['date']}</td><td style="color:var(--accent2)">{r['time']}</td>
          <td><span class="pill present">Present</span></td></tr>"""
          for i,r in enumerate(records,1))
        body = f"""<div class="card">
  <div class="card-header"><h2>All Attendance Records</h2>
    <span class="pill present">{len(records)} Records</span></div>
  <table><thead><tr><th>#</th><th>ID</th><th>Name</th><th>Date</th><th>Time</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody></table></div>"""
    else:
        body = """<div class="card"><div class="card-header"><h2>All Records</h2></div>
  <div class="empty"><div class="icon">&#128452;</div><p>No records yet. Mark attendance first.</p></div></div>"""
    return html_page("All Records", body, "All Records")


# ══════════════════════════════════════════════
# PAGE: STUDENTS
# ══════════════════════════════════════════════
def page_students():
    students = db_query("SELECT student_id,name,date_added FROM students ORDER BY student_id")
    if students:
        rows = ""
        for i, s in enumerate(students, 1):
            cnt = db_query("SELECT COUNT(*) as c FROM attendance WHERE student_id=?", (s['student_id'],))
            days = cnt[0]["c"] if cnt else 0
            rows += f"""<tr><td style="color:var(--muted)">{i}</td>
              <td><img src="/photo?id={s['student_id']}" style="width:36px;height:36px;border-radius:50%;object-fit:cover" onerror="this.style.display='none'"></td>
              <td><span class="pill id">{s['student_id']}</span></td>
              <td><strong><a href="/student_details?id={s['student_id']}" style="color:var(--text);text-decoration:none;border-bottom:1px dashed var(--muted);padding-bottom:2px">{s['name']}</a></strong></td>
              <td style="color:var(--muted)">{s['date_added']}</td>
              <td><span class="pill present">{days} days</span></td>
              <td><button onclick="deleteStudent('{s['student_id']}')" class="btn btn-outline" style="color:var(--red);border-color:var(--red);padding:4px 8px;font-size:11px">Delete</button></td></tr>"""
        body = f"""<div class="card">
  <div class="card-header"><h2>Registered Students</h2>
    <span class="pill present">{len(students)} Students</span></div>
  <table><thead><tr><th>#</th><th>Photo</th><th>ID</th><th>Name</th><th>Enrolled</th><th>Days Present</th><th>Action</th></tr></thead>
  <tbody>{rows}</tbody></table></div>
<script>
async function deleteStudent(sid) {{
  const pwd = prompt("Enter Admin Password to delete student " + sid + ":");
  if (!pwd) return;
  try {{
    const res = await fetch('/api/delete_student', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ student_id: sid, password: pwd }})
    }});
    const data = await res.json();
    if (data.ok) {{ alert("Student deleted!"); location.reload(); }}
    else {{ alert("Error: " + data.error); }}
  }} catch(e) {{ alert("Error: " + e.message); }}
}}
</script>"""
    else:
        body = """<div class="card"><div class="card-header"><h2>Registered Students</h2></div>
  <div class="empty"><div class="icon">&#128100;</div>
  <p>No students yet. <a href="/register" style="color:var(--accent)">Register a student</a>.</p></div></div>"""
    return html_page("Students", body, "Students")


# ══════════════════════════════════════════════
# PAGE: STUDENT DETAILS
# ══════════════════════════════════════════════
def page_student_details(sid):
    if not sid:
        return html_page("Student Details", "<h2>No ID provided</h2>")
        
    s_rows = db_query("SELECT name, date_added FROM students WHERE student_id=? OR CAST(student_id AS INTEGER)=?", (sid, int(sid) if sid.isdigit() else 0))
    if not s_rows:
        return html_page("Student Details", "<h2>Student not found</h2>")
        
    name = s_rows[0]['name']
    enrolled = s_rows[0]['date_added']
    
    # Get all distinct dates when AT LEAST ONE attendance was marked in the whole school
    all_dates = db_query("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
    
    # Get this student's attendance records
    my_att = db_query("SELECT date, time FROM attendance WHERE student_id=? OR CAST(student_id AS INTEGER)=?", (sid, int(sid) if sid.isdigit() else 0))
    my_att_map = {r['date']: r['time'] for r in my_att}
    
    rows = ""
    present_count = 0
    total_count = len(all_dates)
    
    for r in all_dates:
        d = r['date']
        if d in my_att_map:
            status = '<span class="pill present">&#10003; Present</span>'
            time_val = my_att_map[d]
            present_count += 1
        else:
            status = '<span class="pill" style="background:rgba(239,68,68,.15);color:var(--red)">&#10007; Absent</span>'
            time_val = "--"
            
        rows += f"<tr><td>{d}</td><td>{status}</td><td style='color:var(--muted)'>{time_val}</td></tr>"
        
    if not all_dates:
        table = "<p style='padding:24px;text-align:center;color:var(--muted)'>No attendance sessions recorded in the system yet.</p>"
    else:
        table = f"""<table><thead><tr><th>Date</th><th>Status</th><th>Check-in Time</th></tr></thead><tbody>{rows}</tbody></table>"""
        
    body = f"""<div style="max-width:800px;margin:0 auto">
<a href="/students" class="btn btn-outline" style="margin-bottom:20px;font-size:12px;padding:6px 12px">&larr; Back to Students</a>
<div class="card">
  <div class="card-header" style="display:flex;align-items:center;gap:20px;padding:24px">
    <img src="/photo?id={sid}" style="width:80px;height:80px;border-radius:50%;object-fit:cover;border:3px solid var(--border)" onerror="this.style.display='none'">
    <div>
      <h2 style="font-size:24px;margin-bottom:6px">{name}</h2>
      <div style="color:var(--muted);font-size:14px">Student ID: {sid} &bull; Enrolled: {enrolled}</div>
    </div>
  </div>
  <div style="padding:24px;display:flex;gap:20px;background:rgba(255,255,255,.01)">
    <div class="stat-card green" style="flex:1"><div class="stat-label">Present</div><div class="stat-value green">{present_count}</div></div>
    <div class="stat-card" style="flex:1;border-color:rgba(239,68,68,.3)"><div class="stat-label" style="color:var(--red)">Absent</div><div class="stat-value" style="color:var(--red)">{total_count - present_count}</div></div>
    <div class="stat-card purple" style="flex:1"><div class="stat-label">Attendance %</div><div class="stat-value purple">{round((present_count/total_count)*100) if total_count else 0}%</div></div>
  </div>
  <div style="border-top:1px solid var(--border)">
    {table}
  </div>
</div>
</div>"""
    return html_page(f"{name} Details", body, "Students")


# ══════════════════════════════════════════════
# PAGE: REGISTER STUDENT
# ══════════════════════════════════════════════
def page_register():
    body = """
<div style="max-width:900px">
<h2 style="margin-bottom:6px;font-size:20px">Register New Student</h2>
<p style="color:var(--muted);font-size:14px;margin-bottom:24px">
  Enter student details, then capture 3 photos using the live camera or upload existing photos.
  After registration, go to <a href="/train" style="color:var(--accent)">Train Model</a>.
</p>

<div class="card" style="padding:24px">
  <!-- STEP 1: Student Info -->
  <div id="step1">
    <h3 style="margin-bottom:16px;font-size:15px;color:var(--accent)">Step 1 &mdash; Student Information</h3>
    <div class="reg-grid">
      <div class="form-group">
        <label>Student ID (numbers only)</label>
        <input type="number" id="sid" placeholder="e.g. 101" min="1" />
      </div>
      <div class="form-group">
        <label>Full Name</label>
        <input type="text" id="sname" placeholder="e.g. Alice Smith" />
      </div>
    </div>
    <button class="btn btn-primary" onclick="goStep2()">Next &rarr; Capture Photos</button>
  </div>

  <!-- STEP 2: Capture -->
  <div id="step2" style="display:none">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <h3 style="font-size:15px;color:var(--accent)">Step 2 &mdash; Capture Face Photos</h3>
      <span id="student_badge" class="pill id"></span>
    </div>

    <div class="tab-bar">
      <button class="tab active" id="tab_cam" onclick="switchTab('cam')">&#128247; Live Camera</button>
      <button class="tab" id="tab_upload" onclick="switchTab('upload')">&#128194; Upload Photos</button>
    </div>

    <!-- CAMERA TAB -->
    <div id="cam_tab">
      <div class="reg-grid">
        <div>
          <div class="cam-box">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="canvas" width="320" height="240"></canvas>
            <div class="cam-overlay" id="cam_count">0 / 3</div>
          </div>
          <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
            <button class="btn btn-primary" id="btn_start_cam" onclick="startCamera()">&#9654; Start Camera</button>
            <button class="btn btn-success" id="btn_capture" onclick="startGuidedCapture()" style="display:none">
              &#128247; Start Guided Capture (3 photos)
            </button>
            <button class="btn btn-outline" id="btn_stop" onclick="stopCapture()" style="display:none">&#9632; Stop</button>
          </div>
        </div>
        <div>
          <div style="color:var(--muted);font-size:13px;margin-bottom:10px">Capture Progress</div>
          <div class="progress-bar"><div class="progress-fill" id="progress_fill" style="width:0%"></div></div>
          <div id="progress_text" style="font-size:13px;color:var(--muted);margin-bottom:12px">0 of 3 photos captured</div>
          <div id="cam_tips" style="font-size:13px;color:var(--muted);line-height:1.7">
            &#9888; Tips:<br>
            &bull; Face the camera directly<br>
            &bull; Good lighting is important<br>
            &bull; Move slightly between captures<br>
            &bull; Keep 40-80cm from camera<br>
            &bull; Photos auto-save when face detected
          </div>
          <div class="status-box" id="cam_status"></div>
          <div id="done_section" style="display:none;margin-top:16px">
            <div style="color:var(--green);font-weight:700;font-size:15px;margin-bottom:10px">
              &#10003; Registration Complete!
            </div>
            <a href="/train" class="btn btn-success" style="margin-right:8px">&#129504; Train Model Now</a>
            <button class="btn btn-outline" onclick="resetForm()">Add Another Student</button>
          </div>
        </div>
      </div>
    </div>

    <!-- UPLOAD TAB -->
    <div id="upload_tab" style="display:none">
      <div class="upload-zone" onclick="document.getElementById('file_input').click()" id="drop_zone">
        <input type="file" id="file_input" multiple accept="image/*" onchange="handleFiles(this.files)">
        <div style="font-size:40px;margin-bottom:10px">&#128247;</div>
        <div style="font-size:15px;font-weight:600;margin-bottom:6px">Click to select photos</div>
        <div style="font-size:13px;color:var(--muted)">Select 10-50 clear face photos &bull; JPG, PNG accepted</div>
      </div>
      <div class="preview-grid" id="preview_grid"></div>
      <div class="progress-bar" style="margin-top:12px"><div class="progress-fill" id="upload_progress" style="width:0%"></div></div>
      <div id="upload_status_text" style="font-size:13px;color:var(--muted);margin:6px 0">No files selected</div>
      <div class="status-box" id="upload_status"></div>
      <button class="btn btn-success" id="btn_upload" onclick="submitUpload()" style="display:none;margin-top:12px">
        Upload &amp; Register
      </button>
      <div id="upload_done" style="display:none;margin-top:16px">
        <div style="color:var(--green);font-weight:700;font-size:15px;margin-bottom:10px">
          &#10003; Photos uploaded successfully!
        </div>
        <a href="/train" class="btn btn-success" style="margin-right:8px">&#129504; Train Model Now</a>
        <button class="btn btn-outline" onclick="resetForm()">Add Another Student</button>
      </div>
    </div>
  </div>
</div>
</div>

<script>
let stream = null;
let capturing = false;
let captureCount = 0;
let currentSid = '';
let currentName = '';
let uploadFiles = [];

// ── Step 1 → Step 2
async function goStep2() {
  const sid = document.getElementById('sid').value.trim();
  const name = document.getElementById('sname').value.trim();
  if (!sid || isNaN(sid) || parseInt(sid) <= 0) {
    alert('Please enter a valid numeric Student ID (e.g. 101)');
    return;
  }
  if (!name || name.length < 2) {
    alert('Please enter the student name (at least 2 characters)');
    return;
  }
  
  // Check duplicate ID
  try {
    const res = await fetch('/api/check_student?id=' + sid);
    const data = await res.json();
    if (data.exists) {
      alert('Error: Student ID ' + sid + ' is already registered to ' + data.name + '!');
      return;
    }
  } catch(e) {}

  currentSid = sid;
  currentName = name;
  document.getElementById('student_badge').textContent = 'ID: ' + sid + ' | ' + name;
  document.getElementById('step1').style.display = 'none';
  document.getElementById('step2').style.display = 'block';
}

// ── Tab switching
function switchTab(tab) {
  document.getElementById('cam_tab').style.display    = tab === 'cam'    ? 'block' : 'none';
  document.getElementById('upload_tab').style.display = tab === 'upload' ? 'block' : 'none';
  document.getElementById('tab_cam').className    = 'tab' + (tab === 'cam'    ? ' active' : '');
  document.getElementById('tab_upload').className = 'tab' + (tab === 'upload' ? ' active' : '');
  if (tab !== 'cam' && stream) { stopCamera(); }
}

// ── Camera
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: 'user' },
      audio: false
    });
    document.getElementById('video').srcObject = stream;
    document.getElementById('btn_start_cam').style.display = 'none';
    document.getElementById('btn_capture').style.display = 'inline-block';
    showStatus('cam_status', 'Camera ready! Click Auto Capture to begin.', 'info');
  } catch(e) {
    showStatus('cam_status', 'Camera error: ' + e.message + '. Make sure camera permission is granted.', 'error');
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  document.getElementById('btn_start_cam').style.display = 'inline-block';
  document.getElementById('btn_capture').style.display = 'none';
}

async function captureSinglePhoto(stepIndex) {
  const instructions = [
    "Photo 1: Look STRAIGHT at the camera",
    "Photo 2: Turn your head SLIGHTLY RIGHT",
    "Photo 3: Turn your head SLIGHTLY LEFT"
  ];
  showStatus('cam_status', instructions[stepIndex] + " (Detecting face...)", 'info');

  while (capturing && captureCount === stepIndex) {
    const frame = captureFrame();
    try {
      const res = await fetch('/api/capture_frame', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ frame: frame, student_id: currentSid, name: currentName })
      });
      const data = await res.json();
      if (data.ok && data.face_found && data.count > stepIndex) {
        captureCount = data.count;
        updateCamProgress(captureCount);
        showStatus('cam_status', `Photo ${captureCount} captured successfully!`, 'success');
        await sleep(1000); // Wait 1 second before next instruction
        break; // break the loop, move to next step
      } else if (data.ok && !data.face_found) {
        showStatus('cam_status', instructions[stepIndex] + " - No face detected, please adjust position.", 'error');
      }
    } catch(e) { }
    await sleep(300); // 300ms delay between frames
  }
}

async function startGuidedCapture() {
  if (!stream) { alert('Start camera first!'); return; }
  capturing = true;
  captureCount = 0;
  document.getElementById('btn_capture').style.display = 'none';
  document.getElementById('btn_stop').style.display = 'inline-block';

  // Guided 3-step capture
  for (let i = 0; i < 3; i++) {
    if (!capturing) break;
    await captureSinglePhoto(i);
  }

  capturing = false;
  document.getElementById('btn_stop').style.display = 'none';
  if (captureCount >= 3) { onCaptureDone(); }
  else { showStatus('cam_status', 'Capture stopped. ' + captureCount + ' photos saved.', 'info'); }
}

function stopCapture() {
  capturing = false;
}

function captureFrame() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 320, 240);
  return canvas.toDataURL('image/jpeg', 0.85);
}

function updateCamProgress(count) {
  const pct = Math.min(100, Math.round(count / 3 * 100));
  document.getElementById('progress_fill').style.width = pct + '%';
  document.getElementById('progress_text').textContent = count + ' of 3 photos captured';
  document.getElementById('cam_count').textContent = count + ' / 3';
}

function onCaptureDone() {
  stopCamera();
  showStatus('cam_status', 'All 3 photos captured! Student registered successfully.', 'success');
  document.getElementById('btn_capture').style.display = 'none';
  document.getElementById('done_section').style.display = 'block';
}

// ── Upload
function handleFiles(files) {
  uploadFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
  const grid = document.getElementById('preview_grid');
  grid.innerHTML = '';
  uploadFiles.forEach(f => {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    grid.appendChild(img);
  });
  document.getElementById('upload_status_text').textContent =
    uploadFiles.length + ' photo' + (uploadFiles.length !== 1 ? 's' : '') + ' selected';
  document.getElementById('btn_upload').style.display =
    uploadFiles.length >= 3 ? 'inline-block' : 'none';
  if (uploadFiles.length < 3) {
    showStatus('upload_status', 'Please select at least 3 photos for accurate recognition.', 'info');
  } else { hideStatus('upload_status'); }
}

async function submitUpload() {
  if (uploadFiles.length < 1) return;
  document.getElementById('btn_upload').disabled = true;
  showStatus('upload_status', 'Uploading and processing photos...', 'info');
  let saved = 0;
  for (let i = 0; i < uploadFiles.length; i++) {
    const b64 = await fileToB64(uploadFiles[i]);
    try {
      const res = await fetch('/api/capture_frame', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ frame: b64, student_id: currentSid, name: currentName })
      });
      const data = await res.json();
      if (data.ok && data.face_found) { saved = data.count; }
    } catch(e) {}
    const pct = Math.round((i+1) / uploadFiles.length * 100);
    document.getElementById('upload_progress').style.width = pct + '%';
    document.getElementById('upload_status_text').textContent =
      'Processing ' + (i+1) + ' of ' + uploadFiles.length + '...';
  }
  if (saved >= 3) {
    showStatus('upload_status', saved + ' face photos saved successfully!', 'success');
    document.getElementById('upload_done').style.display = 'block';
    document.getElementById('btn_upload').style.display = 'none';
  } else {
    showStatus('upload_status', 'Only ' + saved + ' face photos detected. Use clearer face photos.', 'error');
    document.getElementById('btn_upload').disabled = false;
  }
}

function fileToB64(file) {
  return new Promise((res, rej) => {
    const reader = new FileReader();
    reader.onload = e => res(e.target.result);
    reader.onerror = rej;
    reader.readAsDataURL(file);
  });
}

// ── Helpers
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function showStatus(id, msg, type) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = 'status-box ' + type;
  el.style.display = 'block';
}
function hideStatus(id) { document.getElementById(id).style.display = 'none'; }
function resetForm() { location.reload(); }

// Drag & drop on upload zone
const dz = document.getElementById('drop_zone');
if (dz) {
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.style.borderColor='var(--accent)'; });
  dz.addEventListener('dragleave', () => { dz.style.borderColor='var(--border)'; });
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.style.borderColor = 'var(--border)';
    handleFiles(e.dataTransfer.files);
  });
}
</script>
"""
    return html_page("Register Student", body, "Register Student")


# ══════════════════════════════════════════════
# PAGE: TRAIN MODEL
# ══════════════════════════════════════════════
def page_train():
    # Count dataset students
    ds_students = []
    if os.path.exists(DATASET_PATH):
        for f in os.listdir(DATASET_PATH):
            fp = os.path.join(DATASET_PATH, f)
            if os.path.isdir(fp) and "_" in f:
                imgs = [x for x in os.listdir(fp) if x.lower().endswith(".jpg")]
                ds_students.append({"folder": f, "count": len(imgs)})

    model_exists = os.path.exists(os.path.join(TRAINER_PATH, "model.yml"))
    status = training_status

    rows = ""
    for s in ds_students:
        parts = s["folder"].split("_", 1)
        sid = parts[0] if len(parts) == 2 else "?"
        name = parts[1] if len(parts) == 2 else s["folder"]
        ok = "present" if s["count"] >= 3 else "red"
        rows += f"""<tr><td><span class="pill id">{sid}</span></td>
          <td><strong>{name}</strong></td>
          <td><span class="pill {ok}">{s['count']} photos</span></td>
          <td>{'&#10003; Ready' if s['count'] >= 3 else '&#9888; Need 3+'}</td></tr>"""

    if ds_students:
        table = f"""<div class="card">
  <div class="card-header"><h2>Students Ready to Train</h2>
    <span class="pill present">{len(ds_students)} Found</span></div>
  <table><thead><tr><th>ID</th><th>Name</th><th>Photos</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody></table></div>"""
    else:
        table = """<div class="card"><div class="card-header"><h2>Dataset</h2></div>
  <div class="empty"><div class="icon">&#128192;</div>
  <p>No student data found.</p>
  <p style="margin-top:8px"><a href="/register" style="color:var(--accent)">Register students first</a>.</p>
  </div></div>"""

    model_info = f"""<div class="stat-card {'green' if model_exists else 'yellow'}" style="margin-bottom:20px">
  <div class="stat-label">Model Status</div>
  <div class="stat-value {'green' if model_exists else 'yellow'}" style="font-size:20px;margin-top:8px">
    {'&#10003; Model Ready' if model_exists else '&#9888; Not Trained Yet'}
  </div>
  <div class="stat-sub">{'trainer/model.yml exists' if model_exists else 'Run training to create model'}</div>
</div>"""

    train_status_html = ""
    if status["running"]:
        train_status_html = '<div class="status-box info" style="display:block">Training in progress... please wait.</div>'
    elif status["done"]:
        train_status_html = f'<div class="status-box success" style="display:block">{status["message"]}</div>'

    body = f"""
<h2 style="margin-bottom:6px;font-size:20px">Train Face Recognition Model</h2>
<p style="color:var(--muted);font-size:14px;margin-bottom:24px">
  Train the LBPH model on all registered students' photos. Run this after adding new students.
</p>
{model_info}
{table}
<div style="margin-top:20px">
  {train_status_html}
  <button class="btn btn-success" id="btn_train" onclick="runTraining()"
    {'disabled' if not ds_students or status['running'] else ''}>
    &#129504; Start Training
  </button>
  <span style="margin-left:14px;font-size:13px;color:var(--muted)">
    Training takes 5-20 seconds depending on student count.
  </span>
</div>
<div class="status-box" id="train_status" style="display:{'block' if status['done'] or status['running'] else 'none'};
  margin-top:16px"
  class="{'success' if status['done'] else 'info'}">
</div>

<script>
async function runTraining() {{
  document.getElementById('btn_train').disabled = true;
  document.getElementById('btn_train').textContent = 'Training...';
  const el = document.getElementById('train_status');
  el.style.display = 'block';
  el.className = 'status-box info';
  el.textContent = 'Training LBPH model... please wait (5-20 seconds).';
  try {{
    const res = await fetch('/api/train', {{ method: 'POST' }});
    const data = await res.json();
    if (data.ok) {{
      el.className = 'status-box success';
      el.textContent = 'Training complete! Students: ' + data.students.join(', ');
      document.getElementById('btn_train').textContent = '&#10003; Trained!';
    }} else {{
      el.className = 'status-box error';
      el.textContent = 'Training failed: ' + data.error;
      document.getElementById('btn_train').disabled = false;
      document.getElementById('btn_train').textContent = 'Retry Training';
    }}
  }} catch(e) {{
    el.className = 'status-box error';
    el.textContent = 'Error: ' + e.message;
    document.getElementById('btn_train').disabled = false;
  }}
}}
</script>"""
    return html_page("Train Model", body, "Train Model")


# ══════════════════════════════════════════════
# PAGE: LIVE ATTENDANCE
# ══════════════════════════════════════════════
def page_live():
    body = """
<div style="max-width:900px">
<h2 style="margin-bottom:6px;font-size:20px">Live Web Attendance</h2>
<p style="color:var(--muted);font-size:14px;margin-bottom:24px">
  Start the camera to begin marking attendance automatically. 
  Students recognized will be saved to the database.
</p>

<div class="card" style="padding:24px">
  <div class="reg-grid">
    <div>
      <div class="cam-box">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas" width="320" height="240"></canvas>
        <div class="cam-overlay" id="cam_status_text">Camera Off</div>
      </div>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn btn-primary" id="btn_start_cam" onclick="startCameraAndScan()">&#9654; Start Camera &amp; Scan</button>
        <button class="btn btn-outline" id="btn_stop" onclick="stopAll()" style="display:none">&#9632; Turn Off Camera</button>
      </div>
    </div>
    
    <div>
      <h3 style="font-size:15px;color:var(--accent);margin-bottom:12px">Last Recognized</h3>
      <div class="status-box" id="scan_status" style="display:block;margin-bottom:16px">Waiting to start...</div>
      
      <div id="recognized_list" style="display:flex;flex-direction:column;gap:8px;max-height:200px;overflow-y:auto">
        <!-- populated dynamically -->
      </div>
    </div>
  </div>
</div>
</div>

<script>
let stream = null;
let scanning = false;

async function startCameraAndScan() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: 'user' },
      audio: false
    });
    document.getElementById('video').srcObject = stream;
    document.getElementById('btn_start_cam').style.display = 'none';
    document.getElementById('btn_stop').style.display = 'inline-block';
    
    // Automatically start scanning
    startScanning();
  } catch(e) {
    document.getElementById('scan_status').textContent = 'Camera error: ' + e.message;
    document.getElementById('scan_status').className = 'status-box error';
  }
}

function captureFrame() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 320, 240);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function startScanning() {
  scanning = true;
  document.getElementById('cam_status_text').textContent = 'Scanning...';
  document.getElementById('scan_status').textContent = 'Scanning for faces...';
  document.getElementById('scan_status').className = 'status-box info';

  let lastRecognized = "";

  while (scanning) {
    const frame = captureFrame();
    try {
      const res = await fetch('/api/recognize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ frame: frame })
      });
      const data = await res.json();
      
      if (data.ok && data.name && data.name !== "Unknown") {
        if (data.name !== lastRecognized) {
            lastRecognized = data.name;
            document.getElementById('scan_status').textContent = `✅ ${data.name} marked Present!`;
            document.getElementById('scan_status').className = 'status-box success';
            
            // Add to list
            const list = document.getElementById('recognized_list');
            const item = document.createElement('div');
            item.style = 'padding:10px;background:var(--card);border:1px solid var(--green);border-radius:8px;display:flex;justify-content:space-between;align-items:center';
            item.innerHTML = `<strong>${data.name}</strong><span class="pill present">Present</span>`;
            list.prepend(item);
        }
      } else if (data.ok && data.name === "Unknown") {
        document.getElementById('scan_status').textContent = '❌ Unknown Face (Score: ' + Math.round(data.conf) + ' / Needs < 110)';
        document.getElementById('scan_status').className = 'status-box error';
        lastRecognized = "";
      } else if (!data.ok) {
        document.getElementById('scan_status').textContent = data.error;
        document.getElementById('scan_status').className = 'status-box error';
        scanning = false;
        break;
      }
    } catch(e) {}
    
    // Check twice a second
    await new Promise(r => setTimeout(r, 500));
  }
}

function stopAll() {
  scanning = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  document.getElementById('btn_start_cam').style.display = 'inline-block';
  document.getElementById('btn_stop').style.display = 'none';
  document.getElementById('cam_status_text').textContent = 'Camera Off';
  document.getElementById('scan_status').textContent = 'Camera and scanning turned off.';
  document.getElementById('scan_status').className = 'status-box info';
}
</script>
"""
    return html_page("Live Attendance", body, "Live Attendance")


# ══════════════════════════════════════════════
# API: CAPTURE FRAME
# Receives base64 image, detects face, saves crop
# ══════════════════════════════════════════════
def api_capture_frame(body_bytes):
    try:
        import cv2
        import numpy as np

        data = json.loads(body_bytes.decode("utf-8"))
        frame_b64 = data.get("frame", "")
        sid       = str(data.get("student_id", "")).strip()
        name      = str(data.get("name", "")).strip().replace(" ", "_")

        if not sid or not name or not frame_b64:
            return {"ok": False, "error": "Missing fields"}

        # Validate / sanitize
        if not sid.isdigit():
            return {"ok": False, "error": "Invalid student ID"}

        # Remove data URL prefix
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        # Decode base64 → numpy array
        img_bytes  = base64.b64decode(frame_b64)
        nparr      = np.frombuffer(img_bytes, np.uint8)
        frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"ok": True, "face_found": False, "count": _current_count(sid, name)}

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load detector
        if not os.path.exists(CASCADE_PATH):
            return {"ok": False, "error": "Haarcascade not found. Run python 00_setup.py first."}

        detector = cv2.CascadeClassifier(CASCADE_PATH)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40)
        )

        if len(faces) == 0:
            return {"ok": True, "face_found": False, "count": _current_count(sid, name)}

        # Create folder
        safe_name = name.replace(" ", "_")
        folder = os.path.join(DATASET_PATH, f"{sid}_{safe_name}")
        os.makedirs(folder, exist_ok=True)

        # Count existing images
        existing = len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

        if existing >= 3:
            # Register in DB
            _register_student(sid, name.replace("_", " "))
            return {"ok": True, "face_found": True, "count": existing}

        # Save the largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_crop   = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (100, 100))

        img_path = os.path.join(folder, f"{existing + 1}.jpg")
        cv2.imwrite(img_path, face_resized)

        new_count = existing + 1

        # Register in DB when done
        if new_count >= 3:
            _register_student(sid, name.replace("_", " "))

        return {"ok": True, "face_found": True, "count": new_count}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def _current_count(sid, name):
    safe = name.replace(" ", "_")
    folder = os.path.join(DATASET_PATH, f"{sid}_{safe}")
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])


def _register_student(sid, name):
    from db import setup_db, add_student
    setup_db()
    add_student(sid, name)


# ══════════════════════════════════════════════
# API: TRAIN MODEL
# ══════════════════════════════════════════════
def api_train():
    global training_status
    try:
        import cv2
        import numpy as np
        from PIL import Image

        training_status = {"ok": False, "running": True, "done": False, "message": "", "students": []}

        if not os.path.exists(DATASET_PATH):
            return {"ok": False, "error": "Dataset folder not found. Register students first."}

        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        os.makedirs(TRAINER_PATH, exist_ok=True)

        face_samples = []
        ids          = []
        id_name_map  = {}

        for folder in sorted(os.listdir(DATASET_PATH)):
            fp = os.path.join(DATASET_PATH, folder)
            if not os.path.isdir(fp):
                continue
            parts = folder.split("_", 1)
            if len(parts) != 2:
                continue
            try:
                student_id   = int(parts[0])
                student_name = parts[1]
            except ValueError:
                continue

            id_name_map[student_id] = student_name
            for img_file in os.listdir(fp):
                if not img_file.lower().endswith(".jpg"):
                    continue
                try:
                    pil_img   = Image.open(os.path.join(fp, img_file)).convert("L")
                    img_array = np.array(pil_img, dtype="uint8")
                    face_samples.append(img_array)
                    ids.append(student_id)
                except Exception:
                    pass

        if not face_samples:
            training_status = {"ok": False, "running": False, "done": False, "message": "No images found.", "students": []}
            return {"ok": False, "error": "No training images found."}

        recognizer.train(face_samples, np.array(ids))
        recognizer.write(os.path.join(TRAINER_PATH, "model.yml"))

        with open(os.path.join(TRAINER_PATH, "names.txt"), "w", encoding="utf-8") as f:
            for sid, sname in id_name_map.items():
                f.write(f"{sid}:{sname}\n")

        del face_samples, ids

        student_names = list(id_name_map.values())
        msg = f"Training complete! {len(student_names)} student(s): {', '.join(student_names)}"
        training_status = {"ok": True, "running": False, "done": True, "message": msg, "students": student_names}

        return {"ok": True, "students": student_names, "message": msg}

    except Exception as e:
        training_status = {"ok": False, "running": False, "done": False, "message": str(e), "students": []}
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════
# API: RECOGNIZE FRAME (LIVE ATTENDANCE)
# ══════════════════════════════════════════════
global_recognizer = None
global_names = {}

def load_model():
    global global_recognizer, global_names
    import cv2
    if not os.path.exists(os.path.join(TRAINER_PATH, "model.yml")):
        return False
    global_recognizer = cv2.face.LBPHFaceRecognizer_create()
    global_recognizer.read(os.path.join(TRAINER_PATH, "model.yml"))
    
    global_names = {}
    names_path = os.path.join(TRAINER_PATH, "names.txt")
    if os.path.exists(names_path):
        with open(names_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    sid, sname = line.strip().split(":", 1)
                    global_names[int(sid)] = sname
    return True

def api_recognize(body_bytes):
    try:
        import cv2
        import numpy as np
        
        # Load model if not loaded
        if global_recognizer is None:
            if not load_model():
                return {"ok": False, "error": "Model not trained yet! Please train the model first."}
                
        data = json.loads(body_bytes.decode("utf-8"))
        frame_b64 = data.get("frame", "")
        if not frame_b64:
            return {"ok": False, "error": "No frame"}
            
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
            
        img_bytes  = base64.b64decode(frame_b64)
        nparr      = np.frombuffer(img_bytes, np.uint8)
        frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"ok": True, "name": None}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(CASCADE_PATH)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(40,40))
        
        if len(faces) == 0:
            return {"ok": True, "name": None}
            
        # Only check the largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        
        label, conf = global_recognizer.predict(face_crop)
        
        if conf < 110:
            name = global_names.get(label, "Unknown")
            if name != "Unknown":
                from db import mark_attendance
                mark_attendance(str(label), name)
            return {"ok": True, "name": name, "conf": conf}
        else:
            return {"ok": True, "name": "Unknown", "conf": conf}
            
    except Exception as e:
        return {"ok": False, "error": str(e)}

def api_delete_student(body_bytes):
    try:
        import shutil
        data = json.loads(body_bytes.decode("utf-8"))
        sid = str(data.get("student_id", "")).strip()
        pwd = data.get("password", "")
        
        if pwd != "admin123":
            return {"ok": False, "error": "Incorrect password!"}
            
        if not sid:
            return {"ok": False, "error": "Missing student ID"}
            
        # Delete from DB
        db_execute("DELETE FROM students WHERE student_id=?", (sid,))
        db_execute("DELETE FROM attendance WHERE student_id=?", (sid,))
        
        # Delete dataset folder
        if os.path.exists(DATASET_PATH):
            for folder in os.listdir(DATASET_PATH):
                if folder.startswith(f"{sid}_"):
                    shutil.rmtree(os.path.join(DATASET_PATH, folder), ignore_errors=True)
                    
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ══════════════════════════════════════════════
# HTTP HANDLER
# ══════════════════════════════════════════════
class AttendanceHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # Suppress request logs to keep terminal clean

    def send_html(self, content, code=200):
        enc = content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(enc)))
        self.end_headers()
        self.wfile.write(enc)

    def send_json(self, data, code=200):
        enc = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(enc)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(enc)

    def do_GET(self):
        p = urlparse(self.path).path
        try:
            if p in ("/", "/today"):    self.send_html(page_today())
            elif p == "/all":           self.send_html(page_all())
            elif p == "/students":      self.send_html(page_students())
            elif p == "/register":      self.send_html(page_register())
            elif p == "/live":          self.send_html(page_live())
            elif p == "/train":         self.send_html(page_train())
            elif p == "/api/status":
                self.send_json(training_status)
            elif p == "/api/check_student":
                sid = parse_qs(urlparse(self.path).query).get('id', [''])[0]
                res = db_query("SELECT name FROM students WHERE student_id=?", (sid,))
                if res:
                    self.send_json({"exists": True, "name": res[0]['name']})
                else:
                    self.send_json({"exists": False})
            elif p == "/student_details":
                sid = parse_qs(urlparse(self.path).query).get('id', [''])[0]
                self.send_html(page_student_details(sid))
            elif p == "/logo.png":
                if os.path.exists("logo.png"):
                    with open("logo.png", "rb") as f:
                        img_data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(img_data)))
                    self.end_headers()
                    self.wfile.write(img_data)
                else:
                    self.send_response(404)
                    self.end_headers()
            elif p == "/photo":
                sid = parse_qs(urlparse(self.path).query).get('id', [''])[0]
                photo_served = False
                if sid and os.path.exists(DATASET_PATH):
                    try:
                        sid_int = int(sid)
                    except ValueError:
                        sid_int = None
                        
                    for folder in os.listdir(DATASET_PATH):
                        parts = folder.split("_", 1)
                        if len(parts) == 2 and sid_int is not None:
                            try:
                                folder_id = int(parts[0])
                                if folder_id == sid_int:
                                    fp = os.path.join(DATASET_PATH, folder)
                                    imgs = [x for x in os.listdir(fp) if x.lower().endswith(".jpg")]
                                    if imgs:
                                        img_path = os.path.join(fp, imgs[0])
                                        with open(img_path, "rb") as f:
                                            img_data = f.read()
                                        self.send_response(200)
                                        self.send_header("Content-Type", "image/jpeg")
                                        self.send_header("Content-Length", str(len(img_data)))
                                        self.end_headers()
                                        self.wfile.write(img_data)
                                        photo_served = True
                                    break
                            except ValueError:
                                pass
                if not photo_served:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_html("<h2 style='font-family:sans-serif;padding:40px;color:#ccc'>Page not found</h2>", 404)
        except Exception as e:
            self.send_html(f"<pre style='color:red;padding:20px'>Error: {e}</pre>", 500)

    def do_POST(self):
        p = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length > 0 else b""

        try:
            if p == "/api/capture_frame":
                result = api_capture_frame(body)
                self.send_json(result)

            elif p == "/api/recognize":
                result = api_recognize(body)
                self.send_json(result)

            elif p == "/api/delete_student":
                result = api_delete_student(body)
                self.send_json(result)

            elif p == "/api/train":
                # Run training in background thread
                def train_thread():
                    api_train()
                t = threading.Thread(target=train_thread, daemon=True)
                t.start()
                t.join(timeout=60)  # Wait up to 60s
                if training_status.get("done"):
                    self.send_json(training_status)
                else:
                    self.send_json({"ok": False, "error": training_status.get("message", "Training failed or timed out")})

            else:
                self.send_json({"ok": False, "error": "Unknown endpoint"}, 404)

        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, 500)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        try:
            from db import setup_db
            setup_db()
        except Exception:
            pass

    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(TRAINER_PATH, exist_ok=True)
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    server = HTTPServer(("localhost", PORT), AttendanceHandler)

    print()
    print("=" * 52)
    print("  FACE ATTENDANCE — WEB DASHBOARD + REGISTRATION")
    print("=" * 52)
    print(f"  Open in browser: http://localhost:{PORT}")
    print()
    print("  Pages:")
    print(f"    Today        -> http://localhost:{PORT}/")
    print(f"    All Records  -> http://localhost:{PORT}/all")
    print(f"    Students     -> http://localhost:{PORT}/students")
    print(f"    Register     -> http://localhost:{PORT}/register")
    print(f"    Train Model  -> http://localhost:{PORT}/train")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 52)

    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}/register")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()

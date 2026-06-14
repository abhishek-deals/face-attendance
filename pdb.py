# pdb.py — PostgreSQL-based persistent database for Vercel deployment
# Replaces sqlite3 + filesystem storage with Neon PostgreSQL
# All student records, attendance, face photos and the trained model
# are stored in the cloud database so nothing is ever lost on refresh.
#
# Setup:
#   1. Create a free account at https://neon.tech
#   2. Create a new project and database
#   3. Copy the connection string (starts with postgresql://...)
#   4. In Vercel dashboard → Settings → Environment Variables
#      add: DATABASE_URL = <your neon connection string>
# ─────────────────────────────────────────────────────────────

import os
import psycopg2
import psycopg2.extras
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_conn():
    """Get a PostgreSQL connection from the DATABASE_URL env var."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Please add your Neon PostgreSQL connection string in Vercel settings."
        )
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def setup_db():
    """Create all tables if they don't exist. Safe to call multiple times."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id  TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    date_added  TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id          SERIAL PRIMARY KEY,
                    student_id  TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    date        TEXT NOT NULL,
                    time        TEXT NOT NULL
                )
            """)
            # Store face photos as binary blobs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_photos (
                    id          SERIAL PRIMARY KEY,
                    student_id  TEXT NOT NULL,
                    photo_index INTEGER NOT NULL,
                    photo_data  BYTEA NOT NULL,
                    UNIQUE(student_id, photo_index)
                )
            """)
            # Store the trained recognition model as binary
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_model (
                    id          SERIAL PRIMARY KEY,
                    model_data  BYTEA NOT NULL,
                    names_data  TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
            """)
        conn.commit()
        print("PostgreSQL database ready.")
    finally:
        conn.close()


def add_student(sid, name):
    """Register a student. Returns True if inserted, False if duplicate."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            today = datetime.now().strftime("%Y-%m-%d")
            cur.execute("""
                INSERT INTO students (student_id, name, date_added)
                VALUES (%s, %s, %s)
                ON CONFLICT (student_id) DO NOTHING
            """, (str(sid), str(name), today))
            inserted = cur.rowcount > 0
        conn.commit()
        return inserted
    finally:
        conn.close()


def mark_attendance(sid, name):
    """Mark attendance once per student per day. Returns True if newly marked."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            today = datetime.now().strftime("%Y-%m-%d")
            now_time = datetime.now().strftime("%H:%M:%S")
            cur.execute("""
                SELECT id FROM attendance
                WHERE student_id = %s AND date = %s
            """, (str(sid), today))
            if cur.fetchone() is None:
                cur.execute("""
                    INSERT INTO attendance (student_id, name, date, time)
                    VALUES (%s, %s, %s, %s)
                """, (str(sid), str(name), today, now_time))
                conn.commit()
                return True
            return False
    finally:
        conn.close()


def get_today_report():
    """Fetch today's attendance as a list of dicts."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            today = datetime.now().strftime("%Y-%m-%d")
            cur.execute("""
                SELECT student_id, name, date, time
                FROM attendance WHERE date = %s ORDER BY time ASC
            """, (today,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_all_report():
    """Fetch all attendance records as a list of dicts."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT student_id, name, date, time
                FROM attendance ORDER BY date DESC, time DESC
            """)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_students_list():
    """Return all registered students as a list of dicts."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT student_id, name, date_added
                FROM students ORDER BY student_id ASC
            """)
            return [{"id": r["student_id"], "name": r["name"], "date_added": r["date_added"]}
                    for r in cur.fetchall()]
    finally:
        conn.close()


def save_face_photo(sid, photo_index, photo_bytes):
    """Save a face photo to the database (overwrite if exists)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO face_photos (student_id, photo_index, photo_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (student_id, photo_index) DO UPDATE
                SET photo_data = EXCLUDED.photo_data
            """, (str(sid), photo_index, psycopg2.Binary(photo_bytes)))
        conn.commit()
    finally:
        conn.close()


def get_face_photo_count(sid):
    """Return how many photos are stored for a student."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM face_photos WHERE student_id = %s
            """, (str(sid),))
            return cur.fetchone()[0]
    finally:
        conn.close()


def get_all_face_photos():
    """Return all face photos as list of (student_id, photo_bytes) tuples."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT f.student_id, s.name, f.photo_data
                FROM face_photos f
                JOIN students s ON f.student_id = s.student_id
                ORDER BY f.student_id, f.photo_index
            """)
            return [(r[0], r[1], bytes(r[2])) for r in cur.fetchall()]
    finally:
        conn.close()


def get_first_face_photo(sid):
    """Return the first photo bytes for a student (for profile display)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT photo_data FROM face_photos
                WHERE student_id = %s ORDER BY photo_index ASC LIMIT 1
            """, (str(sid),))
            row = cur.fetchone()
            return bytes(row[0]) if row else None
    finally:
        conn.close()


def save_model(model_bytes, names_dict):
    """Save the trained model to the database."""
    import json
    names_json = json.dumps(names_dict)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Keep only the latest model
            cur.execute("DELETE FROM face_model")
            cur.execute("""
                INSERT INTO face_model (model_data, names_data, created_at)
                VALUES (%s, %s, %s)
            """, (psycopg2.Binary(model_bytes), names_json,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    finally:
        conn.close()


def load_model():
    """Load the trained model from the database. Returns (model_bytes, names_dict) or None."""
    import json
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_data, names_data FROM face_model
                ORDER BY id DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return bytes(row[0]), json.loads(row[1])
            return None, None
    finally:
        conn.close()


def delete_student(sid):
    """Delete a student and all their data."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM face_photos WHERE student_id = %s", (str(sid),))
            cur.execute("DELETE FROM attendance WHERE student_id = %s", (str(sid),))
            cur.execute("DELETE FROM students WHERE student_id = %s", (str(sid),))
        conn.commit()
    finally:
        conn.close()


def export_csv(date=None):
    """Export attendance for a date to a CSV string."""
    import csv, io
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT student_id, name, date, time FROM attendance
                WHERE date = %s ORDER BY time ASC
            """, (date,))
            rows = cur.fetchall()
        if not rows:
            return None
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Student ID", "Name", "Date", "Time"])
        writer.writerows(rows)
        return buf.getvalue()
    finally:
        conn.close()


# Alias for compatibility
def export_excel(date=None):
    return export_csv(date)

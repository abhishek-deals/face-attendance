# pdb.py — Dual-backend persistent database for Vercel deployment
#
# Automatically chooses the right backend:
#   • If DATABASE_URL env var is set → uses Neon PostgreSQL (persistent)
#   • If DATABASE_URL is NOT set     → falls back to SQLite in /tmp (ephemeral per-instance)
#
# The SQLite fallback means the app works on Vercel even without a Neon DB.
# Data in SQLite /tmp is ephemeral (lost on cold start), but students/photos/model
# registered within the same Vercel instance session will work correctly.
#
# To get full persistence, add DATABASE_URL to Vercel environment variables:
#   1. Create a free account at https://neon.tech
#   2. Create a new project and database
#   3. Copy the connection string (starts with postgresql://...)
#   4. Vercel dashboard → Settings → Environment Variables
#      add: DATABASE_URL = <your neon connection string>
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import sqlite3
from datetime import datetime as dt, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
class datetime(dt):
    @classmethod
    def now(cls, tz=None):
        return dt.now(IST)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# SQLite DB path — /tmp is writable on Vercel serverless
_is_vercel = bool(os.environ.get("VERCEL", ""))
_default_db = "/tmp/faceattendance.db" if _is_vercel else "attendance.db"
_SQLITE_PATH = os.environ.get("DB_PATH", _default_db)

# ─────────────────────────────────────────────────────────────────────────────
# Backend selection helper
# ─────────────────────────────────────────────────────────────────────────────

def _use_postgres():
    return bool(DATABASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sqlite_conn():
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_setup():
    conn = _sqlite_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                date_added  TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL,
                name        TEXT NOT NULL,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_photos (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL,
                photo_index INTEGER NOT NULL,
                photo_data  BLOB NOT NULL,
                UNIQUE(student_id, photo_index)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_model (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_data  BLOB NOT NULL,
                names_data  TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.commit()
        print("SQLite database ready (fallback mode — no DATABASE_URL set).")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pg_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def _pg_setup():
    import psycopg2
    conn = _pg_conn()
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_photos (
                    id          SERIAL PRIMARY KEY,
                    student_id  TEXT NOT NULL,
                    photo_index INTEGER NOT NULL,
                    photo_data  BYTEA NOT NULL,
                    UNIQUE(student_id, photo_index)
                )
            """)
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


# ─────────────────────────────────────────────────────────────────────────────
# Public API — same interface regardless of backend
# ─────────────────────────────────────────────────────────────────────────────

def setup_db():
    """Create all tables if they don't exist. Safe to call multiple times."""
    if _use_postgres():
        _pg_setup()
    else:
        _sqlite_setup()


# ─────────────────────────────────────────────────────────────────────────────

def add_student(sid, name):
    """Register a student. Returns True if inserted, False if duplicate."""
    today = datetime.now().strftime("%Y-%m-%d")
    if _use_postgres():
        import psycopg2
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
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
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR IGNORE INTO students (student_id, name, date_added)
                VALUES (?, ?, ?)
            """, (str(sid), str(name), today))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def mark_attendance(sid, name):
    """Mark attendance once per student per day. Returns True if newly marked."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    if _use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM attendance WHERE student_id = %s AND date = %s
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
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id FROM attendance WHERE student_id = ? AND date = ?
            """, (str(sid), today))
            if cur.fetchone() is None:
                cur.execute("""
                    INSERT INTO attendance (student_id, name, date, time)
                    VALUES (?, ?, ?, ?)
                """, (str(sid), str(name), today, now_time))
                conn.commit()
                return True
            return False
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def get_today_report():
    """Fetch today's attendance as a list of dicts."""
    today = datetime.now().strftime("%Y-%m-%d")
    if _use_postgres():
        import psycopg2.extras
        conn = _pg_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT student_id, name, date, time
                    FROM attendance WHERE date = %s ORDER BY time ASC
                """, (today,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT student_id, name, date, time
                FROM attendance WHERE date = ? ORDER BY time ASC
            """, (today,))
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def get_all_report():
    """Fetch all attendance records as a list of dicts."""
    if _use_postgres():
        import psycopg2.extras
        conn = _pg_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT student_id, name, date, time
                    FROM attendance ORDER BY date DESC, time DESC
                """)
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT student_id, name, date, time
                FROM attendance ORDER BY date DESC, time DESC
            """)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def get_students_list():
    """Return all registered students as a list of dicts."""
    if _use_postgres():
        import psycopg2.extras
        conn = _pg_conn()
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
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT student_id, name, date_added
                FROM students ORDER BY student_id ASC
            """)
            return [{"id": r["student_id"], "name": r["name"], "date_added": r["date_added"]}
                    for r in cur.fetchall()]
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def save_face_photo(sid, photo_index, photo_bytes):
    """Save a face photo to the database (overwrite if exists)."""
    if _use_postgres():
        import psycopg2
        conn = _pg_conn()
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
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO face_photos (student_id, photo_index, photo_data)
                VALUES (?, ?, ?)
            """, (str(sid), photo_index, photo_bytes))
            conn.commit()
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def get_face_photo_count(sid):
    """Return how many photos are stored for a student."""
    if _use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM face_photos WHERE student_id = %s
                """, (str(sid),))
                return cur.fetchone()[0]
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM face_photos WHERE student_id = ?", (str(sid),))
            return cur.fetchone()[0]
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def get_all_face_photos():
    """Return all face photos as list of (student_id, name, photo_bytes) tuples."""
    if _use_postgres():
        conn = _pg_conn()
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
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT f.student_id, s.name, f.photo_data
                FROM face_photos f
                JOIN students s ON f.student_id = s.student_id
                ORDER BY f.student_id, f.photo_index
            """)
            return [(r[0], r[1], bytes(r[2])) for r in cur.fetchall()]
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def get_first_face_photo(sid):
    """Return the first photo bytes for a student (for profile display)."""
    if _use_postgres():
        conn = _pg_conn()
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
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT photo_data FROM face_photos
                WHERE student_id = ? ORDER BY photo_index ASC LIMIT 1
            """, (str(sid),))
            row = cur.fetchone()
            return bytes(row[0]) if row else None
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def save_model(model_bytes, names_dict):
    """Save the trained model to the database."""
    names_json = json.dumps(names_dict)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if _use_postgres():
        import psycopg2
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM face_model")
                cur.execute("""
                    INSERT INTO face_model (model_data, names_data, created_at)
                    VALUES (%s, %s, %s)
                """, (psycopg2.Binary(model_bytes), names_json, now))
            conn.commit()
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM face_model")
            cur.execute("""
                INSERT INTO face_model (model_data, names_data, created_at)
                VALUES (?, ?, ?)
            """, (model_bytes, names_json, now))
            conn.commit()
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load the trained model from the database. Returns (model_bytes, names_dict) or (None, None)."""
    if _use_postgres():
        conn = _pg_conn()
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
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
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


# ─────────────────────────────────────────────────────────────────────────────

def delete_student(sid):
    """Delete a student and all their data."""
    if _use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM face_photos WHERE student_id = %s", (str(sid),))
                cur.execute("DELETE FROM attendance WHERE student_id = %s", (str(sid),))
                cur.execute("DELETE FROM students WHERE student_id = %s", (str(sid),))
            conn.commit()
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM face_photos WHERE student_id = ?", (str(sid),))
            cur.execute("DELETE FROM attendance WHERE student_id = ?", (str(sid),))
            cur.execute("DELETE FROM students WHERE student_id = ?", (str(sid),))
            conn.commit()
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────

def export_csv(date=None):
    """Export attendance for a date to a CSV string."""
    import csv, io
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if _use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT student_id, name, date, time FROM attendance
                    WHERE date = %s ORDER BY time ASC
                """, (date,))
                rows = cur.fetchall()
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT student_id, name, date, time FROM attendance
                WHERE date = ? ORDER BY time ASC
            """, (date,))
            rows = cur.fetchall()
        finally:
            conn.close()

    if not rows:
        return None
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Student ID", "Name", "Date", "Time"])
    writer.writerows(rows)
    return buf.getvalue()


# Alias for compatibility
def export_excel(date=None):
    return export_csv(date)

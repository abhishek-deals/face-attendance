# db.py — Lightweight Database Module
# Uses only built-in sqlite3 + pandas + openpyxl + csv
# No heavy dependencies. Zero RAM overhead.
#
# FUNCTIONS:
#   setup_db()           → create tables
#   add_student()        → register student
#   mark_attendance()    → mark once per day
#   get_today_report()   → today's DataFrame
#   get_all_report()     → full history DataFrame
#   get_students_list()  → all registered students
#   export_excel()       → save .xlsx to attendance/
#   export_csv()         → save .csv  to attendance/
# ──────────────────────────────────────────────

import sqlite3
import os
from datetime import datetime

# Pandas is only imported where needed to save memory at startup
# openpyxl is imported only inside export_excel()

DB_PATH = "attendance.db"
ATTENDANCE_DIR = "attendance"


# ──────────────────────────────────────────────
# FUNCTION 1: setup_db()
# Creates tables if they don't already exist.
# Safe to call multiple times (idempotent).
# ──────────────────────────────────────────────
def setup_db():
    """Initialize the SQLite database and create tables."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Students master table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                date_added  TEXT NOT NULL
            )
        """)

        # Attendance log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL,
                name        TEXT NOT NULL,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL
            )
        """)

        conn.commit()
        print("Database ready.")

    except sqlite3.Error as e:
        print(f"[DB ERROR] setup_db: {e}")

    finally:
        if conn:
            conn.close()


# ──────────────────────────────────────────────
# FUNCTION 2: add_student(sid, name)
# Registers a new student. Skips if already exists.
# Returns True if inserted, False if duplicate/error.
# ──────────────────────────────────────────────
def add_student(sid, name):
    """Insert student into DB. Ignores duplicates silently."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        # INSERT OR IGNORE: won't crash on duplicate primary key
        cursor.execute("""
            INSERT OR IGNORE INTO students
            (student_id, name, date_added)
            VALUES (?, ?, ?)
        """, (str(sid), str(name), today))

        conn.commit()

        if cursor.rowcount > 0:
            print(f"[DB] Student registered: {name} (ID: {sid})")
            return True
        else:
            print(f"[DB] Student already exists: {name} (ID: {sid})")
            return False

    except sqlite3.Error as e:
        print(f"[DB ERROR] add_student: {e}")
        return False

    finally:
        if conn:
            conn.close()


# ──────────────────────────────────────────────
# FUNCTION 3: mark_attendance(sid, name)
# Marks attendance only once per day per student.
# Returns True  → newly marked (first time today)
# Returns False → already marked today (duplicate)
# ──────────────────────────────────────────────
def mark_attendance(sid, name):
    """Mark attendance. One entry per student per day."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%H:%M:%S")

        # Check if already marked today
        cursor.execute("""
            SELECT id FROM attendance
            WHERE student_id = ? AND date = ?
        """, (str(sid), today))

        row = cursor.fetchone()

        if row is None:
            # Not yet marked — insert new record
            cursor.execute("""
                INSERT INTO attendance
                (student_id, name, date, time)
                VALUES (?, ?, ?, ?)
            """, (str(sid), str(name), today, now_time))
            conn.commit()
            return True   # Newly marked

        else:
            # Already marked today
            return False  # Duplicate — do nothing

    except sqlite3.Error as e:
        print(f"[DB ERROR] mark_attendance: {e}")
        return False

    finally:
        if conn:
            conn.close()


# ──────────────────────────────────────────────
# FUNCTION 4: get_today_report()
# Returns today's attendance as a pandas DataFrame.
# ──────────────────────────────────────────────
def get_today_report():
    """Fetch today's attendance records as a DataFrame."""
    import pandas as pd  # Imported here to reduce startup RAM

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        today = datetime.now().strftime("%Y-%m-%d")

        df = pd.read_sql_query("""
            SELECT student_id, name, date, time
            FROM attendance
            WHERE date = ?
            ORDER BY time ASC
        """, conn, params=(today,))

        return df

    except Exception as e:
        # Catches both sqlite3.Error and pandas errors
        print(f"[DB ERROR] get_today_report: {e}")
        return pd.DataFrame()

    finally:
        if conn:
            conn.close()


# ──────────────────────────────────────────────
# FUNCTION 5: get_all_report()
# Returns all attendance records as a DataFrame.
# Most recent dates/times shown first.
# ──────────────────────────────────────────────
def get_all_report():
    """Fetch all attendance records ordered by newest first."""
    import pandas as pd

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql_query("""
            SELECT student_id, name, date, time
            FROM attendance
            ORDER BY date DESC, time DESC
        """, conn)

        return df

    except Exception as e:
        # Catches both sqlite3.Error and pandas errors
        print(f"[DB ERROR] get_all_report: {e}")
        return pd.DataFrame()

    finally:
        if conn:
            conn.close()


# ──────────────────────────────────────────────
# FUNCTION 6: export_excel(date=None)
# Exports attendance report to an Excel file.
# If date is None → uses today's date.
# Returns the file path string on success.
# ──────────────────────────────────────────────
def export_excel(date=None):
    """Export attendance report to Excel (.xlsx) file."""
    import pandas as pd
    # openpyxl is the engine used by pandas for xlsx

    # Determine which date to export
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Ensure output folder exists
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql_query("""
            SELECT student_id AS "Student ID",
                   name       AS "Name",
                   date       AS "Date",
                   time       AS "Time"
            FROM attendance
            WHERE date = ?
            ORDER BY time ASC
        """, conn, params=(date,))

        if df.empty:
            print(f"[EXPORT] No records found for date: {date}")
            return None

        file_path = os.path.join(ATTENDANCE_DIR, f"report_{date}.xlsx")

        # Write Excel using openpyxl engine
        df.to_excel(file_path, index=False, engine="openpyxl")

        print(f"[EXPORT] Saved: {file_path} ({len(df)} records)")
        return file_path

    except Exception as e:
        print(f"[DB ERROR] export_excel: {e}")
        return None

    finally:
        if conn:
            conn.close()


# ──────────────────────────────────────────────
# FUNCTION 7: export_csv(date=None)
# Exports attendance as a daily CSV file.
# attendance/ folder is described as 'daily CSV files'.
# Uses built-in csv module — no extra install needed.
# If date is None → uses today's date.
# Returns the file path string on success.
# ──────────────────────────────────────────────
def export_csv(date=None):
    """Export attendance report to CSV file (built-in csv module)."""
    import csv  # Built-in — zero extra install

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT student_id, name, date, time
            FROM attendance
            WHERE date = ?
            ORDER BY time ASC
        """, (date,))

        rows = cursor.fetchall()

        if not rows:
            print(f"[EXPORT] No records found for date: {date}")
            return None

        file_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date}.csv")

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header row
            writer.writerow(["Student ID", "Name", "Date", "Time"])
            # Data rows
            writer.writerows(rows)

        print(f"[EXPORT] CSV saved: {file_path} ({len(rows)} records)")
        return file_path

    except Exception as e:
        print(f"[DB ERROR] export_csv: {e}")
        return None

    finally:
        if conn:
            conn.close()


# ──────────────────────────────────────────────
# FUNCTION 8: get_students_list()
# Returns all registered students as a list of dicts.
# Used by other scripts to display enrolled students.
# ──────────────────────────────────────────────
def get_students_list():
    """Return all registered students from the students table."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT student_id, name, date_added
            FROM students
            ORDER BY student_id ASC
        """)

        rows = cursor.fetchall()
        students = [
            {"id": r[0], "name": r[1], "date_added": r[2]}
            for r in rows
        ]
        return students

    except sqlite3.Error as e:
        print(f"[DB ERROR] get_students_list: {e}")
        return []

    finally:
        if conn:
            conn.close()

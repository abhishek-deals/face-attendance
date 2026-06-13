# 04_report.py — Attendance Report Viewer & Exporter
#
# Pure text-based menu — no GUI, no Tkinter, no lag.
# Works perfectly even on 4GB RAM laptops.
#
# MENU OPTIONS:
#   1. Today's attendance
#   2. All records (full history)
#   3. Export today to Excel (.xlsx)
#   4. Export specific date to Excel
#   5. Export today to CSV
#   6. View registered students
#   7. Exit
# ──────────────────────────────────────────────────────

import os
import sys
from datetime import datetime

# Import DB functions
try:
    from db import (
        setup_db,
        get_today_report,
        get_all_report,
        get_students_list,
        export_excel,
        export_csv
    )
except ImportError as e:
    print(f"ERROR: Cannot import db.py — {e}")
    print("Make sure db.py is in the same folder as this script.")
    sys.exit(1)


# ──────────────────────────────────────────────
# Initialize DB (creates tables if not present)
# ──────────────────────────────────────────────
setup_db()


# ──────────────────────────────────────────────
# Helper: Print separator line
# ──────────────────────────────────────────────
def separator():
    print("-" * 50)


# ──────────────────────────────────────────────
# ACTION 1: Show today's attendance
# ──────────────────────────────────────────────
def show_today():
    today = datetime.now().strftime("%Y-%m-%d")
    print()
    separator()
    print(f"  TODAY'S ATTENDANCE  ({today})")
    separator()

    df = get_today_report()

    if df is None or df.empty:
        print("  No attendance records found for today.")
        print("  Run 03_attendance.py to mark attendance.")
    else:
        # Print formatted table
        print(df.to_string(index=False))
        separator()
        print(f"  Total students present: {len(df)}")

    separator()


# ──────────────────────────────────────────────
# ACTION 2: Show ALL records
# ──────────────────────────────────────────────
def show_all():
    print()
    separator()
    print("  ALL ATTENDANCE RECORDS  (newest first)")
    separator()

    df = get_all_report()

    if df is None or df.empty:
        print("  No attendance records found in database.")
        print("  Run 03_attendance.py to mark attendance.")
    else:
        # If records are many, show a count first
        total = len(df)

        print(df.to_string(index=False))
        separator()
        print(f"  Total records: {total}")

        # Show unique student count
        if "name" in df.columns:
            unique_students = df["name"].nunique()
            print(f"  Unique students: {unique_students}")

        # Show date range
        if "date" in df.columns:
            dates = df["date"].unique()
            print(f"  Dates covered  : {len(dates)} day(s)")

    separator()


# ──────────────────────────────────────────────
# ACTION 3: Export today to Excel
# ──────────────────────────────────────────────
def export_today():
    today = datetime.now().strftime("%Y-%m-%d")
    print()
    separator()
    print(f"  EXPORTING REPORT  ({today})")
    separator()

    # Verify openpyxl is available before attempting export
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  ERROR: openpyxl is not installed.")
        print("  Install it with: pip install openpyxl")
        separator()
        return

    file_path = export_excel(date=today)

    if file_path:
        abs_path = os.path.abspath(file_path)
        print("  [OK] Export successful!")
        print(f"  File: {abs_path}")
    else:
        print("  [X] Export failed.")
        print(f"  No records found for {today}.")
        print("  Make sure attendance was marked today.")

    separator()


# ──────────────────────────────────────────────
# ACTION: Export a specific date to Excel
# ──────────────────────────────────────────────
def export_specific_date():
    print()
    separator()
    print("  EXPORT SPECIFIC DATE")
    separator()

    # Get date input from user
    while True:
        date_input = input(
            "  Enter date (YYYY-MM-DD) or press Enter for today: "
        ).strip()

        if date_input == "":
            date_input = datetime.now().strftime("%Y-%m-%d")
            break

        # Validate format
        try:
            datetime.strptime(date_input, "%Y-%m-%d")
            break
        except ValueError:
            print("  [!] Invalid format. Use YYYY-MM-DD (e.g. 2025-01-15)")

    print(f"  Exporting records for: {date_input}")

    # Verify openpyxl is available
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  ERROR: openpyxl is not installed.")
        print("  Install it with: pip install openpyxl")
        separator()
        return

    file_path = export_excel(date=date_input)

    if file_path:
        abs_path = os.path.abspath(file_path)
        print(f"  [OK] Export successful!")
        print(f"  File: {abs_path}")
    else:
        print(f"  [X] No records found for {date_input}.")

    separator()


# ──────────────────────────────────────────────
# ACTION 5: Export today to CSV
# Uses built-in csv module — no openpyxl needed
# ──────────────────────────────────────────────
def export_csv_today():
    today = datetime.now().strftime("%Y-%m-%d")
    print()
    separator()
    print(f"  EXPORTING CSV  ({today})")
    separator()

    file_path = export_csv(date=today)

    if file_path:
        abs_path = os.path.abspath(file_path)
        print(f"  [OK] CSV export successful!")
        print(f"  File: {abs_path}")
        print(f"  Open with: Notepad, Excel, or any spreadsheet app")
    else:
        print(f"  [X] No records found for {today}.")
        print("  Make sure attendance was marked today.")

    separator()


# ──────────────────────────────────────────────
# ACTION 6: View all registered students
# Shows every student enrolled in the system.
# ──────────────────────────────────────────────
def show_students():
    print()
    separator()
    print("  REGISTERED STUDENTS")
    separator()

    students = get_students_list()

    if not students:
        print("  No students registered yet.")
        print("  Run 01_collect.py to add students.")
    else:
        print(f"  {'ID':<8} {'Name':<25} {'Date Added'}")
        print(f"  {'-'*7} {'-'*24} {'-'*12}")
        for s in students:
            print(f"  {s['id']:<8} {s['name']:<25} {s['date_added']}")
        separator()
        print(f"  Total registered: {len(students)} student(s)")

    separator()


# ──────────────────────────────────────────────
# MAIN MENU LOOP
# ──────────────────────────────────────────────
def main():
    print()
    print("=" * 50)
    print("     FACE ATTENDANCE — REPORT VIEWER")
    print("=" * 50)

    while True:
        print()
        print("  MENU:")
        print("    1. Today's attendance")
        print("    2. All records (full history)")
        print("    3. Export today to Excel (.xlsx)")
        print("    4. Export specific date to Excel")
        print("    5. Export today to CSV")
        print("    6. View registered students")
        print("    7. Exit")
        print()

        choice = input("  Enter choice (1-7): ").strip()

        if choice == "1":
            show_today()

        elif choice == "2":
            show_all()

        elif choice == "3":
            export_today()

        elif choice == "4":
            export_specific_date()

        elif choice == "5":
            export_csv_today()

        elif choice == "6":
            show_students()

        elif choice == "7":
            print()
            print("  Goodbye!")
            print()
            break

        else:
            print()
            print("  [!] Invalid choice. Please enter 1 through 7.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()

@echo off
REM ============================================================
REM  install.bat — One-click package installer
REM  Face Recognition Attendance System
REM  Installs packages ONE BY ONE to avoid RAM overload
REM  Designed for: AMD Athlon 4GB RAM laptops / Python 3.13
REM
REM  IMPORTANT: Do NOT pin opencv to 4.8.0.76 on Python 3.13
REM  That version uses NumPy 1.x which is incompatible with
REM  Python 3.13's bundled NumPy 2.x → causes ImportError.
REM  This batch file installs the latest compatible versions.
REM ============================================================

echo.
echo  ============================================
echo   FACE ATTENDANCE SYSTEM — PACKAGE INSTALLER
echo  ============================================
echo.
echo  Installing packages one by one to avoid RAM overload...
echo  This will take 3-7 minutes. Please wait.
echo.

echo  [1/5] Installing opencv-python (base)...
pip install opencv-python
echo.

echo  [2/5] Installing opencv-contrib-python (LBPH needed)...
pip install opencv-contrib-python
echo.

echo  [3/5] Installing pandas (reports)...
pip install pandas
echo.

echo  [4/5] Installing Pillow (image loading)...
pip install Pillow
echo.

echo  [5/5] Installing openpyxl (Excel export)...
pip install openpyxl
echo.

echo  ============================================
echo   All packages installed!
echo  ============================================
echo.
echo  Now run: python 00_setup.py
echo  (Downloads haarcascade XML and sets up database)
echo.
pause

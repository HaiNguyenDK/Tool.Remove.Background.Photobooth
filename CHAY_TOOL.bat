@echo off
REM ============================================================
REM   NHAP DUP file nay de chay Tool Tach Nen Photobooth.
REM   Uu tien chay ban .exe (khong can Python).
REM   Neu chua build exe -> chay bang Python 3.11.
REM ============================================================
cd /d "%~dp0"

if exist "dist\TachNenPhotobooth\TachNenPhotobooth.exe" (
    start "" "dist\TachNenPhotobooth\TachNenPhotobooth.exe"
    exit /b 0
)

echo Khong tim thay exe, dang chay bang Python 3.11...
py -3.11 app.py
if errorlevel 1 (
    echo.
    echo [Loi] Can Python 3.11 + thu vien. Cai bang:
    echo     py -3.11 -m pip install -r requirements.txt
    pause
)

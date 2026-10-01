@echo off
REM ====== Khoi dong Tool Tach Nen AI ======
cd /d "%~dp0"
echo Dang khoi dong Tool Tach Nen AI...
py -3.11 app.py
if errorlevel 1 (
    echo.
    echo [Loi] Khong chay duoc. Hay cai dependencies:
    echo     py -3.11 -m pip install -r requirements.txt
    pause
)

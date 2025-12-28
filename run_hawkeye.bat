@echo off
echo ============================================================
echo   Volleyball Hawk-Eye Tactical Intelligence System
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show ultralytics >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_hawkeye.txt
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Running Volleyball Hawk-Eye System...
echo.
echo Input:  C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\input_videos\Video1.mp4
echo Weights: C:\Users\xghostrider\Downloads\best(2).pt
echo Output: C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\output_tactical.mp4
echo.

python volleyball_hawkeye.py

echo.
echo [3/3] Processing complete!
echo.
echo Check output file: output_tactical.mp4
echo.
pause

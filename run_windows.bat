@echo off
title Road Damage Detector
echo ============================================
echo   Road Damage Detector - first-time setup
echo ============================================
echo.

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Installing dependencies (first time can take a few minutes)...
pip install -r requirements.txt

echo.
echo Starting the app... a browser window will open.
echo Press CTRL+C in this window to stop.
echo.
streamlit run app.py

pause

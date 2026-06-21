#!/bin/bash
echo "============================================"
echo "  Road Damage Detector - first-time setup"
echo "============================================"
echo

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies (first time can take a few minutes)..."
pip install -r requirements.txt

echo
echo "Starting the app... a browser window will open."
echo "Press CTRL+C to stop."
echo
streamlit run app.py

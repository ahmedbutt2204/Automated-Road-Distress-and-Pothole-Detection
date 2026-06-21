# 🛣️ Road Damage & Pothole Detector

Upload a road photo → the trained YOLOv11 model finds cracks and potholes
and shows them clearly with labels and confidence.

Detects 4 damage types: D00 Longitudinal crack, D10 Transverse crack,
D20 Alligator crack, D40 Pothole.

---

## ✅ Easiest way to run

You only need **Python installed**. Then:

- **Windows:** double-click **`run_windows.bat`**
- **Mac / Linux:** open a terminal here and run `bash run_mac_linux.sh`

It installs everything the first time, then opens the app at **http://localhost:8501**.
(First run downloads PyTorch + Ultralytics — give it a few minutes.)

---

## 🧰 Manual way

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🖱️ How to use

1. Click **Browse files** and choose one or more road images.
2. The app shows the annotated image, summary counts (including potholes found),
   a simple list of detections with confidence, and a download button.

---

## 📁 Files

Keep all model files in the same folder as `app.py`.

*Information Technology University — Department of Computer Science / AI*

# 🛣️ Automated Road Distress and Pothole Detection

A deep-learning system that automatically detects **road cracks and potholes** from a road image.
Built with **YOLOv11** and a simple **Streamlit** web app — upload a road photo and get the damage
highlighted with boxes, types, and confidence scores.

---

## ✨ Features

- Detects **4 types of road damage** in a single image
- Clean web interface — just upload an image and view results
- Works with **any image format** and **multiple images** at once
- Downloadable annotated images
- Fast, real-time detection (under 20 ms per image)

### Damage types detected

| Code | Damage type        |
|------|--------------------|
| D00  | Longitudinal crack |
| D10  | Transverse crack   |
| D20  | Alligator crack    |
| D40  | Pothole            |

---

## 📊 Results

| Metric        | Score |
|---------------|-------|
| mAP@0.5       | 0.89  |
| F1-Score      | 0.88  |
| Precision     | 0.885 |
| Recall        | 0.875 |
| Pothole (D40) | 0.93  |

Model: **YOLOv11m** · trained on the **RDD2022** dataset (India, China, Japan).

---

## 📁 Project structure

```
road-damage-detector/
├── app.py              # the Streamlit web app
├── best.pt             # trained model (cracks)
├── potholebest.pt      # trained model (potholes)
├── requirements.txt    # Python libraries needed
├── run_windows.bat     # one-click run for Windows
├── run_mac_linux.sh    # one-click run for Mac / Linux
├── .streamlit/         # app config
└── README.md
```

---

## ⚙️ Setup — How to run the project

You need **Python 3.9 or newer** installed. Check with:

```bash
python --version
```

### Step 1 — Get the project

Either download the ZIP from GitHub (green **Code** button → **Download ZIP**) and extract it,
**or** clone it:

```bash
git clone https://github.com/ahmedbutt2204/Automated-Road-Distress-and-Pothole-Detection.git
cd Automated-Road-Distress-and-Pothole-Detection
```

### Step 2 — Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install the required libraries

```bash
pip install -r requirements.txt
```

> The first install downloads PyTorch and Ultralytics, so give it a few minutes.

### Step 4 — Run the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at **http://localhost:8501**

---

## 🖱️ How to use

1. Click **Browse files** and select one or more road images.
2. The app shows each image with coloured boxes around the damage, the damage type, and a
   confidence score.
3. See the summary counts (images, detections, potholes found).
4. Click **Download annotated image** to save the result.

> Keep `best.pt` and `potholebest.pt` in the same folder as `app.py`.

---

## ⚡ Quick run (no commands)

If Python is already installed, you can skip the terminal:

- **Windows:** double-click **`run_windows.bat`**
- **Mac / Linux:** run `bash run_mac_linux.sh`

These set everything up and open the app automatically.

---

## 🧠 Tech stack

- **YOLOv11** (Ultralytics) — object detection
- **PyTorch** — deep-learning backend
- **Streamlit** — web interface
- **RDD2022** — Road Damage Detection 2022 dataset

---

## 👥 Authors

- **Ahmed Nadeem** — BSCS23185

*Information Technology University — Department of Computer Science / AI*

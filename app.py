"""
Road Distress Detection — Streamlit frontend
Upload road images; the trained YOLOv11 model detects cracks and potholes
and shows clean, annotated results.

Run:  streamlit run app.py
"""

import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_MAIN = "best.pt"
MODEL_POTHOLE = "potholebest.pt"
CRACK_CONF = 0.15
POTHOLE_CONF = 0.25
IMGSZ = 768

CRACK_CODES = ["D00", "D10", "D20"]
NAMES = {"D00": "Longitudinal crack", "D10": "Transverse crack",
         "D20": "Alligator crack", "D40": "Pothole"}
COLORS = {"D00": "#E63946", "D10": "#F3722C", "D20": "#577590", "D40": "#1F9D74"}
FALLBACK = "#888780"

st.set_page_config(page_title="Road Distress Detection", page_icon="🛣️",
                   layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
:root{
  --ink:#16222E; --muted:#6A7682; --line:#E7EBEE; --brand:#17242F;
  --accent:#E0913B; --green:#1F9D74; --soft:#F7F9FB;
}
.block-container{ max-width:1140px; padding-top:1.6rem; padding-bottom:2rem; }
html, body, [class*="css"]{ font-family:'Segoe UI',Helvetica,Arial,sans-serif; color:var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"]{ display:none; }

/* top bar */
.topbar{ display:flex; align-items:center; justify-content:space-between;
  padding:0 0 18px; border-bottom:1px solid var(--line); margin-bottom:22px; }
.brand{ display:flex; align-items:center; gap:15px; }
.logo{ width:46px; height:46px; border-radius:12px; background:var(--brand);
  position:relative; overflow:hidden; box-shadow:0 4px 12px rgba(22,34,46,.18); flex:none; }
.logo::before{ content:""; position:absolute; left:50%; bottom:8px; transform:translateX(-50%);
  width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
  border-bottom:26px solid rgba(224,145,59,.92); }
.logo::after{ content:""; position:absolute; left:50%; bottom:10px; transform:translateX(-50%);
  width:2.4px; height:20px;
  background:repeating-linear-gradient(#17242F,#17242F 3px,transparent 3px,transparent 6px); }
.brand-title{ font-size:21px; font-weight:700; letter-spacing:-.3px; line-height:1.1; }
.brand-sub{ font-size:13px; color:var(--muted); margin-top:2px; }
.badges{ display:flex; gap:8px; }
.badge{ font-size:12px; font-weight:600; color:var(--muted); background:var(--soft);
  border:1px solid var(--line); padding:6px 12px; border-radius:8px; white-space:nowrap; }
.badge.dot::before{ content:""; display:inline-block; width:7px; height:7px; border-radius:50%;
  background:var(--green); margin-right:7px; vertical-align:middle; }

/* file uploader */
[data-testid="stFileUploaderDropzone"]{ background:var(--soft); border:1.6px dashed #CDD6DD;
  border-radius:14px; padding:20px; }

/* summary strip */
.summary{ display:flex; align-items:center; background:var(--brand);
  border-radius:14px; padding:18px 8px; margin:6px 0 22px; }
.s-item{ flex:1; text-align:center; }
.s-num{ font-size:27px; font-weight:800; color:#fff; line-height:1; }
.s-num.acc{ color:var(--accent); } .s-num.grn{ color:#54C99E; }
.s-lbl{ font-size:12px; color:#9FB0BD; margin-top:6px; }
.s-sep{ width:1px; height:34px; background:rgba(255,255,255,.12); }

/* results */
.sec-head{ font-size:17px; font-weight:700; margin:4px 0 14px; }
.rname{ font-size:15px; font-weight:700; margin:0 0 12px; }
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.chip{ font-size:12.5px; font-weight:600; color:#fff; padding:5px 12px; border-radius:8px; }
.tbl{ width:100%; border-collapse:collapse; font-size:13px; }
.tbl th{ text-align:left; color:var(--muted); font-weight:600; font-size:11px;
  text-transform:uppercase; letter-spacing:.4px; padding:0 0 8px; border-bottom:1px solid var(--line); }
.tbl td{ padding:9px 0; border-bottom:1px solid #F1F4F6; }
.dotc{ display:inline-block; width:9px; height:9px; border-radius:3px; margin-right:8px; vertical-align:middle; }
.conf{ font-weight:700; }
.empty{ color:var(--green); font-weight:600; font-size:14px; padding-top:10px; }
.legend-row{ display:flex; align-items:center; gap:9px; margin:7px 0; font-size:13.5px; }
.legend-dot{ width:13px; height:13px; border-radius:4px; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the detection model…")
def load_models(main_path, pothole_path):
    return YOLO(main_path), YOLO(pothole_path)


def get_font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_results(pil_img, detections):
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, round(min(w, h) / 280))
    font = get_font(max(15, round(min(w, h) / 40)))
    for d in detections:
        color = COLORS.get(d["code"], FALLBACK)
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"{NAMES.get(d['code'], d['code'])}  {d['conf']:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 8)
        draw.rectangle([x1, ly, x1 + tw + 14, ly + th + 8], fill=color)
        draw.text((x1 + 7, ly + 4), label, fill="white", font=font)
    return img


def run_detection(models, pil_img):
    main_model, pothole_model = models
    out = []
    r1 = main_model.predict(pil_img, conf=CRACK_CONF, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
    names = r1.names
    for b in r1.boxes:
        code = names.get(int(b.cls), str(int(b.cls)))
        if code not in CRACK_CODES:
            continue
        out.append({"code": code, "conf": float(b.conf),
                    "box": [int(v) for v in b.xyxy[0].tolist()]})
    r2 = pothole_model.predict(pil_img, conf=POTHOLE_CONF, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
    for b in r2.boxes:
        out.append({"code": "D40", "conf": float(b.conf),
                    "box": [int(v) for v in b.xyxy[0].tolist()]})
    out.sort(key=lambda d: d["conf"], reverse=True)
    return out


def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Top bar
# --------------------------------------------------------------------------
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="logo"></div>
    <div>
      <div class="brand-title">Road Distress Detection</div>
      <div class="brand-sub">Automated crack &amp; pothole analysis</div>
    </div>
  </div>
  <div class="badges">
    <span class="badge dot">Model ready</span>
    <span class="badge">YOLOv11 &middot; RDD2022</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Load models
# --------------------------------------------------------------------------
missing = [m for m in (MODEL_MAIN, MODEL_POTHOLE) if not Path(m).exists()]
if missing:
    st.error("Missing model file(s): " + ", ".join(missing) +
             ". Keep best.pt and potholebest.pt next to app.py.")
    st.stop()
models = load_models(MODEL_MAIN, MODEL_POTHOLE)

# --------------------------------------------------------------------------
# Sidebar (legend / about)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("##### Damage types")
    for code in ["D00", "D10", "D20", "D40"]:
        st.markdown(
            f'<div class="legend-row"><span class="legend-dot" '
            f'style="background:{COLORS[code]}"></span>{NAMES[code]} ({code})</div>',
            unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: YOLOv11 trained on RDD2022.\nUse clear, street-level road photos for best results.")

# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------
files = st.file_uploader("Upload road image(s) — any image format",
                         type=None, accept_multiple_files=True)
if not files:
    st.info("Upload one or more road images to begin.")
    st.stop()

# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
all_counts, total, processed = {}, 0, []
with st.spinner("Analysing road images…"):
    for f in files:
        try:
            pil = Image.open(f); pil.load()
        except Exception:
            st.warning(f"'{f.name}' is not a readable image — skipped.")
            continue
        dets = run_detection(models, pil)
        annotated = draw_results(pil, dets)
        processed.append((f.name, annotated, dets))
        total += len(dets)
        for d in dets:
            all_counts[d["code"]] = all_counts.get(d["code"], 0) + 1

if not processed:
    st.error("No valid images were processed.")
    st.stop()

# --------------------------------------------------------------------------
# Summary strip
# --------------------------------------------------------------------------
potholes = all_counts.get("D40", 0)
st.markdown(f"""
<div class="summary">
  <div class="s-item"><div class="s-num">{len(processed)}</div><div class="s-lbl">Images analysed</div></div>
  <div class="s-sep"></div>
  <div class="s-item"><div class="s-num">{total}</div><div class="s-lbl">Total detections</div></div>
  <div class="s-sep"></div>
  <div class="s-item"><div class="s-num grn">{potholes}</div><div class="s-lbl">Potholes found</div></div>
  <div class="s-sep"></div>
  <div class="s-item"><div class="s-num acc">{len(all_counts)}</div><div class="s-lbl">Damage types</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-head">Detection Results</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Per-image results
# --------------------------------------------------------------------------
for name, annotated, dets in processed:
    with st.container(border=True):
        left, right = st.columns([1.7, 1], gap="large")
        with left:
            st.image(annotated, use_container_width=True)
            st.download_button("Download annotated image", data=pil_to_bytes(annotated),
                               file_name=f"detected_{name}.png", mime="image/png", key=f"dl_{name}")
        with right:
            st.markdown(f'<div class="rname">{name}</div>', unsafe_allow_html=True)
            if not dets:
                st.markdown('<div class="empty">No damage detected.</div>', unsafe_allow_html=True)
            else:
                counts = {}
                for d in dets:
                    counts[d["code"]] = counts.get(d["code"], 0) + 1
                chips = "".join(
                    f'<span class="chip" style="background:{COLORS.get(c, FALLBACK)}">'
                    f'{NAMES.get(c, c)} &times; {n}</span>' for c, n in counts.items())
                rows = "".join(
                    f'<tr><td><span class="dotc" style="background:{COLORS.get(d["code"], FALLBACK)}">'
                    f'</span>{NAMES.get(d["code"], d["code"])}</td>'
                    f'<td class="conf">{round(d["conf"]*100)}%</td></tr>' for d in dets)
                st.markdown(
                    f'<div class="chips">{chips}</div>'
                    f'<table class="tbl"><tr><th>Damage type</th><th>Confidence</th></tr>{rows}</table>',
                    unsafe_allow_html=True)
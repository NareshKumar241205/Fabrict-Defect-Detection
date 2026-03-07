"""
FabricQA — Automated Fabric Defect Detection
==============================================
Upload an image → Unified Processor routes to correct engines
→ results categorized into Group I (Structure) and Group II (Stitch).
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import base64
import tempfile
from io import BytesIO
from datetime import datetime
from fpdf import FPDF

from inspectors import texture_inspector, spectral_inspector, seam_inspector, edge_inspector
from inspectors.unified_processor import unified_processor

# ──────────────────────────────────────────────
# DEFECT CLASSIFICATION MAP (10-Type Taxonomy)
# ──────────────────────────────────────────────
# Group I: Fabric Structure defects (physical / pattern anomalies)
GROUP_I_TYPES = {
    "Missing Thread", "Slub", "Oil Stain",
    "Hole", "Tear", "Snag",
}
# Group II: Stitch Quality defects (seam-related)
GROUP_II_TYPES = {
    "Skip Stitch", "Broken Stitch", "Run-off Stitch",
    "Crooked Stitch", "Pucker",
}
# Structural = physically damaging; Surface = visual/textural
STRUCTURAL_TYPES = {
    "Missing Thread", "Hole", "Tear", "Snag",
    "Skip Stitch", "Broken Stitch", "Run-off Stitch", "Crooked Stitch",
}
SURFACE_TYPES = {
    "Slub", "Oil Stain", "Pucker",
}

def classify_defect(defect_type: str) -> str:
    """Categorize a defect as Structural or Surface."""
    if defect_type in STRUCTURAL_TYPES:
        return "Structural"
    if defect_type in SURFACE_TYPES:
        return "Surface"
    # fallback heuristic
    low = defect_type.lower()
    if any(k in low for k in ("tear", "hole", "cut", "stitch", "break", "weave", "thread", "snag")):
        return "Structural"
    return "Surface"

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FabricQA · Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# DARK MODE CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary:    #0a0e17;
    --bg-secondary:  #111827;
    --bg-card:       rgba(17, 24, 39, 0.7);
    --bg-glass:      rgba(255, 255, 255, 0.03);
    --border:        rgba(255, 255, 255, 0.06);
    --border-glow:   rgba(99, 102, 241, 0.15);
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --accent:        #818cf8;
    --accent-bright: #a78bfa;
    --accent-glow:   rgba(129, 140, 248, 0.15);
    --success:       #34d399;
    --success-bg:    rgba(52, 211, 153, 0.1);
    --danger:        #f87171;
    --danger-bg:     rgba(248, 113, 113, 0.1);
    --warning:       #fbbf24;
    --warning-bg:    rgba(251, 191, 36, 0.1);
    --structural-clr:#f87171;
    --structural-bg: rgba(248, 113, 113, 0.08);
    --surface-clr:   #fbbf24;
    --surface-bg:    rgba(251, 191, 36, 0.08);
    --gradient-2:    linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    --gradient-3:    linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    --shadow:        0 4px 24px rgba(0, 0, 0, 0.4);
    --shadow-glow:   0 0 30px rgba(99, 102, 241, 0.08);
    --radius:        16px;
    --radius-sm:     10px;
}

html, body, [class*="stApp"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
div[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }

/* ── Hero ── */
.hero-banner {
    background: var(--gradient-3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-glow);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner h1 {
    margin: 0; font-size: 1.8rem; font-weight: 800;
    background: var(--gradient-2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.5px;
}
.hero-banner p {
    margin: 6px 0 0; font-size: 0.85rem;
    color: var(--text-secondary); font-weight: 400; letter-spacing: 0.3px;
}

/* ── Section Titles ── */
.sec-title {
    font-size: 0.7rem; font-weight: 700; color: var(--accent);
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 2rem 0 0.8rem; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Metric cards ── */
.metrics-row { display: flex; gap: 16px; margin: 1.2rem 0; }
.m-card {
    flex: 1;
    background: var(--bg-card); backdrop-filter: blur(12px);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 1.2rem 1.4rem; text-align: center;
    box-shadow: var(--shadow);
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.m-card:hover { border-color: var(--border-glow); transform: translateY(-1px); }
.m-card .m-val {
    font-size: 1.8rem; font-weight: 800;
    background: var(--gradient-2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.m-card .m-lbl {
    font-size: 0.65rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 4px; font-weight: 600;
}

/* ── Category headers ── */
.cat-hdr {
    display: flex; align-items: center; gap: 10px;
    padding: 0.7rem 1.2rem; border-radius: var(--radius-sm);
    margin-bottom: 0.8rem; font-weight: 700; font-size: 0.95rem;
}
.cat-structural {
    background: var(--structural-bg);
    border: 1px solid rgba(248,113,113,0.2);
    color: var(--structural-clr);
}
.cat-surface {
    background: var(--surface-bg);
    border: 1px solid rgba(251,191,36,0.2);
    color: var(--surface-clr);
}

/* ── Badges ── */
.badge-pass {
    display: inline-block; background: var(--success-bg); color: var(--success);
    font-weight: 700; padding: 4px 18px; border-radius: 20px; font-size: 0.9rem;
    border: 1px solid rgba(52,211,153,0.2); letter-spacing: 0.5px;
}
.badge-fail {
    display: inline-block; background: var(--danger-bg); color: var(--danger);
    font-weight: 700; padding: 4px 18px; border-radius: 20px; font-size: 0.9rem;
    border: 1px solid rgba(248,113,113,0.2); letter-spacing: 0.5px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 4rem 2rem;
    background: var(--bg-card); backdrop-filter: blur(12px);
    border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; opacity: 0.6; }
.empty-state p { color: var(--text-muted); margin-top: 8px; font-size: 0.9rem; }

/* ── Streamlit overrides ── */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important; backdrop-filter: blur(12px) !important;
    border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
}
label, .stSlider label, .stRadio label, .stFileUploader label,
div[data-testid="stWidgetLabel"] p, div[data-testid="stMarkdownContainer"] p,
.stSelectbox label, .stMultiSelect label { color: var(--text-secondary) !important; }
div[data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }
div[data-testid="stSlider"] div[role="slider"] { background-color: var(--accent) !important; }

.stButton > button {
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
    font-size: 0.78rem !important; letter-spacing: 0.2px !important;
    transition: all 0.25s ease !important; border: 1px solid var(--border) !important;
    padding: 0.55rem 1rem !important;
}
.stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
    background: var(--bg-glass) !important; color: var(--text-secondary) !important;
}
.stButton > button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
    background: var(--accent-glow) !important; color: var(--accent) !important;
    border-color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
    background: var(--gradient-2) !important; color: white !important;
    border: none !important; box-shadow: 0 2px 12px rgba(99,102,241,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(99,102,241,0.5) !important; transform: translateY(-1px);
}

div[data-testid="stFileUploader"] section {
    background: var(--bg-glass) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
div[data-testid="stDataFrame"] { border-radius: var(--radius-sm) !important; overflow: hidden; }
.stDownloadButton > button {
    background: var(--bg-glass) !important; color: var(--accent) !important;
    border: 1px solid var(--accent) !important; border-radius: var(--radius-sm) !important;
}
.stDownloadButton > button:hover { background: var(--accent-glow) !important; }
div[data-testid="stAlert"] {
    background: var(--bg-card) !important; border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}
div[data-testid="stProgress"] > div > div > div { background: var(--gradient-2) !important; }
details {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
details summary { color: var(--text-secondary) !important; }
button[data-baseweb="tab"] { color: var(--text-muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
section[data-testid="stSidebar"] { background: var(--bg-secondary) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HISTORY HELPER
# ──────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "inspection_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    history = history[-200:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def clone_buffer(f):
    if f is None:
        return None
    f.seek(0)
    return BytesIO(f.read())


def validate_image(img_file):
    if img_file is None:
        return None, None
    img_file.seek(0)
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_file.seek(0)
    if img is None:
        return None, "❌ Could not decode image. File may be corrupt."
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return None, f"❌ Image too small ({w}×{h}). Min 100×100."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 5:
        return None, "⚠️ Image appears blank or solid color."
    return img, None


def deduplicate_defects(defects, iou_threshold=0.5):
    """Remove overlapping defects using Non-Maximum Suppression.
    
    When two defect boxes overlap >= iou_threshold, the one with
    lower confidence is dropped.
    """
    if len(defects) <= 1:
        return defects

    def _conf(d):
        c = d.get("Confidence", "0%")
        try: return int(str(c).replace("%", "").strip())
        except: return 0

    def _iou(a, b):
        ax1, ay1 = a.get("bbox_x", 0), a.get("bbox_y", 0)
        ax2 = ax1 + a.get("bbox_w", 0)
        ay2 = ay1 + a.get("bbox_h", 0)
        bx1, by1 = b.get("bbox_x", 0), b.get("bbox_y", 0)
        bx2 = bx1 + b.get("bbox_w", 0)
        by2 = by1 + b.get("bbox_h", 0)

        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / max(union, 1)

    # Sort by confidence descending
    sorted_defs = sorted(defects, key=_conf, reverse=True)
    keep = []

    for d in sorted_defs:
        suppressed = False
        for kept in keep:
            if _iou(d, kept) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)

    return keep


def decode_image(img_file):
    """Read an image file into a BGR numpy array."""
    img_file.seek(0)
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img_file.seek(0)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ── PDF REPORT ──
def generate_pdf_report(base_img, structural_overlay, surface_overlay, all_defects, verdict, s_count, f_count):
    """Generate a PDF report with images, defect table, and verdict."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Helper: save BGR image to temp JPEG, return path
    def _save_temp(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return tmp.name

    tmp_files = []

    try:
        # Page 1: Header + Verdict + Images
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 12, "FabricQA - Inspection Report", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Verdict
        pdf.set_font("Helvetica", "B", 16)
        color = (220, 53, 69) if verdict == "FAIL" else (40, 167, 69)
        pdf.set_text_color(*color)
        pdf.cell(0, 10, f"Verdict: {verdict}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Total Defects: {s_count + f_count}  |  Structural: {s_count}  |  Surface: {f_count}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Images
        img_w = 58
        for label, img in [("Original", base_img), ("Structural", structural_overlay), ("Surface", surface_overlay)]:
            tmp = _save_temp(img)
            tmp_files.append(tmp)

        pdf.set_font("Helvetica", "B", 10)
        x_start = 10
        for i, label in enumerate(["Original", "Structural Defects", "Surface Defects"]):
            x = x_start + i * 63
            pdf.set_xy(x, pdf.get_y())
            pdf.cell(img_w, 5, label)

        y_img = pdf.get_y() + 6
        for i, tmp in enumerate(tmp_files):
            pdf.image(tmp, x=x_start + i * 63, y=y_img, w=img_w)

        # Page 2: Defect Table
        if all_defects:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Defect Details", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

            # Table header
            headers = ["#", "Category", "Type", "Area (px)", "Confidence"]
            col_w = [10, 30, 50, 25, 25]
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(240, 240, 240)
            for j, h in enumerate(headers):
                pdf.cell(col_w[j], 7, h, border=1, fill=True)
            pdf.ln()

            # Table rows
            pdf.set_font("Helvetica", "", 8)
            for idx, d in enumerate(all_defects[:50], 1):  # cap at 50 rows
                pdf.cell(col_w[0], 6, str(idx), border=1)
                pdf.cell(col_w[1], 6, d.get("Category", ""), border=1)
                pdf.cell(col_w[2], 6, d.get("Type", "")[:30], border=1)
                pdf.cell(col_w[3], 6, str(d.get("Area (px)", "")), border=1)
                pdf.cell(col_w[4], 6, str(d.get("Confidence", "")), border=1)
                pdf.ln()

        return pdf.output()

    finally:
        for f in tmp_files:
            try: os.unlink(f)
            except: pass


# ── BEFORE/AFTER COMPARISON SLIDER ──
def render_comparison_slider(original_bgr, annotated_bgr):
    """Render an interactive before/after slider using HTML/CSS/JS."""
    def _to_b64(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()

    b64_before = _to_b64(original_bgr)
    b64_after = _to_b64(annotated_bgr)

    uid = f"slider_{id(original_bgr)}"

    html = f"""
    <div id="{uid}" style="position:relative;width:100%;max-width:700px;margin:0 auto;overflow:hidden;
         border-radius:12px;border:1px solid rgba(255,255,255,0.1);user-select:none;-webkit-user-select:none;">
      <img src="data:image/jpeg;base64,{b64_after}" style="width:100%;display:block;" />
      <div id="{uid}_clip" style="position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden;">
        <img src="data:image/jpeg;base64,{b64_before}" style="width:100%;height:100%;object-fit:cover;
             min-width:200%;max-width:none;" id="{uid}_bimg" />
      </div>
      <div id="{uid}_handle" style="position:absolute;top:0;left:50%;width:3px;height:100%;
           background:rgba(255,255,255,0.9);cursor:ew-resize;z-index:10;transform:translateX(-50%);">
        <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
             background:rgba(255,255,255,0.95);border-radius:50%;width:32px;height:32px;
             display:flex;align-items:center;justify-content:center;font-size:14px;
             box-shadow:0 2px 8px rgba(0,0,0,0.3);">⇔</div>
      </div>
      <div style="position:absolute;top:8px;left:8px;background:rgba(0,0,0,0.6);color:#fff;
           padding:2px 8px;border-radius:4px;font-size:11px;">Original</div>
      <div style="position:absolute;top:8px;right:8px;background:rgba(0,0,0,0.6);color:#fff;
           padding:2px 8px;border-radius:4px;font-size:11px;">Annotated</div>
    </div>
    <script>
    (function(){{
      var container = document.getElementById("{uid}");
      var clip = document.getElementById("{uid}_clip");
      var handle = document.getElementById("{uid}_handle");
      var bimg = document.getElementById("{uid}_bimg");
      var dragging = false;
      function update(x) {{
        var rect = container.getBoundingClientRect();
        var pct = Math.max(0, Math.min(1, (x - rect.left) / rect.width));
        clip.style.width = (pct * 100) + "%";
        handle.style.left = (pct * 100) + "%";
        bimg.style.minWidth = (100 / Math.max(pct, 0.01)) + "%";
      }}
      container.addEventListener("mousedown", function(e) {{ dragging = true; update(e.clientX); }});
      window.addEventListener("mousemove", function(e) {{ if (dragging) update(e.clientX); }});
      window.addEventListener("mouseup", function() {{ dragging = false; }});
      container.addEventListener("touchstart", function(e) {{ dragging = true; update(e.touches[0].clientX); }});
      container.addEventListener("touchmove", function(e) {{ if (dragging) {{ update(e.touches[0].clientX); e.preventDefault(); }} }});
      container.addEventListener("touchend", function() {{ dragging = false; }});
    }})();
    </script>
    """
    st.components.v1.html(html, height=500)


# ── DEFECT HEATMAP (BATCH) ──
def generate_defect_heatmap(all_defects, canvas_size=600):
    """Generate an aggregate heatmap showing where defects cluster."""
    heatmap = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    if not all_defects:
        return None

    for d in all_defects:
        bx = d.get("bbox_x", 0)
        by = d.get("bbox_y", 0)
        bw = d.get("bbox_w", 0)
        bh = d.get("bbox_h", 0)
        if bw <= 0 or bh <= 0:
            continue
        # Normalize to canvas coordinates (assume max 2000px original)
        scale = canvas_size / 2000.0
        x1 = max(0, min(canvas_size - 1, int(bx * scale)))
        y1 = max(0, min(canvas_size - 1, int(by * scale)))
        x2 = max(0, min(canvas_size, int((bx + bw) * scale)))
        y2 = max(0, min(canvas_size, int((by + bh) * scale)))
        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] += 1.0

    if np.max(heatmap) == 0:
        return None

    # Blur and normalize
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return colored


# ── FEEDBACK LOOP ──
FEEDBACK_FILE = "feedback_log.json"

def load_feedback():
    """Load user feedback from JSON file."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except: pass
    return []

def save_feedback(entry):
    """Append a feedback entry to JSON file."""
    data = load_feedback()
    data.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


def draw_category_overlay(base_img, defects, color_bgr, label_prefix):
    """Draw bounding boxes for a list of defects onto a copy of the image."""
    overlay = base_img.copy()
    img_h, img_w = overlay.shape[:2]

    for d in defects:
        bx = d.get("bbox_x")
        by = d.get("bbox_y")
        bw = d.get("bbox_w")
        bh = d.get("bbox_h")

        if bx is None or by is None or bw is None or bh is None:
            continue

        x1 = max(0, int(bx))
        y1 = max(0, int(by))
        x2 = min(img_w, int(bx + bw))
        y2 = min(img_h, int(by + bh))

        if x2 <= x1 or y2 <= y1:
            continue

        # Thick bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 3)

        # Semi-transparent fill only for small/medium boxes
        box_area = (x2 - x1) * (y2 - y1)
        img_area = img_h * img_w
        if box_area < img_area * 0.15:
            sub = overlay[y1:y2, x1:x2]
            if sub.size > 0:
                fill = np.full_like(sub, color_bgr, dtype=np.uint8)
                cv2.addWeighted(fill, 0.18, sub, 0.82, 0, sub)

        # Label tag above box
        lbl = d.get("Type", "Defect")
        conf = d.get("Confidence", "")
        tag = f"{lbl} ({conf})" if conf else lbl

        font_scale = max(0.4, min(0.7, bw / 200))
        thickness = 1 if font_scale < 0.55 else 2
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Background rectangle for text readability
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(overlay, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color_bgr, -1)
        cv2.putText(overlay, tag, (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🔬 FabricQA</h1>
    <p>Upload a fabric image · Get a complete defect report · Structural vs Surface classification</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TAB NAVIGATION  (Inspect / Batch / History)
# ──────────────────────────────────────────────
tab_inspect, tab_batch, tab_history = st.tabs(["🔍  Inspect", "📦  Batch", "📊  History"])


# ══════════════════════════════════════════════
# TAB 1 — SINGLE INSPECTION
# ══════════════════════════════════════════════
with tab_inspect:
    st.markdown('<div class="sec-title">Upload Fabric Image</div>', unsafe_allow_html=True)

    col_up, col_cfg = st.columns([3, 1])

    with col_up:
        with st.container(border=True):
            src = st.radio("Source", ["Upload Image", "Live Camera"], horizontal=True, key="src_radio")
            if src == "Upload Image":
                img_file = st.file_uploader("Choose a fabric image", type=["jpg", "png", "bmp"], key="single_upload")
            else:
                img_file = st.camera_input("Capture", key="cam")

    with col_cfg:
        with st.container(border=True):
            st.markdown("**⚙️ Settings**")
            sensitivity = st.slider("Detection sensitivity", 1.0, 5.0, 2.5, 0.1,
                                    help="Higher = more sensitive, may produce false positives",
                                    key="sens_slider")
            conf_threshold = st.slider("Confidence threshold (%)", 0, 80, 30, 5,
                                       help="Defects below this confidence are filtered out",
                                       key="conf_slider")
            remove_shadows_ui = st.checkbox("Remove Shadows (Illumination Correction)", value=False,
                                            help="Enable this if uneven lighting is causing false positive defects",
                                            key="shadow_toggle")

        with st.container(border=True):
            st.markdown("**🖼️ Reference Compare (Golden Image)**")
            st.caption("Upload a known-good sample for SSIM-based comparison")
            ref_file = st.file_uploader("Golden image", type=["jpg", "png", "bmp"], key="ref_upload")
            if ref_file:
                st.image(ref_file, caption="Golden Reference", width=150)

    # ── PROCESS ──
    if img_file is not None:
        orig_img, validation_error = validate_image(img_file)
        if validation_error:
            st.warning(validation_error)
        elif orig_img is not None:
            try:
                progress = st.progress(0, text="⏳ Starting unified inspection…")

                # ── UNIFIED PROCESSOR: Intelligent Routing Pipeline ──
                progress.progress(0.1, text="🔍  Pre-classifying image region…")
                buf = clone_buffer(img_file)
                
                progress.progress(0.2, text="🧠  Routing to detection engines…")
                ref_buf = clone_buffer(ref_file) if ref_file else None
                result = unified_processor.process(buf, sensitivity=sensitivity, mode="full", remove_shadows=remove_shadows_ui, ref_buffer=ref_buf)
                
                progress.progress(0.9, text="📊  Compiling results…")
                all_defects = result["defects"]
                viz_maps = result["viz_maps"]
                routing_info = result["routing_info"]
                
                progress.progress(1.0, text="✅  Inspection complete!")

                # ── CLASSIFY & FILTER ──
                for d in all_defects:
                    d["Category"] = classify_defect(d.get("Type", ""))

                # Parse confidence string "42%" → 42, filter below threshold
                def _conf_val(d):
                    c = d.get("Confidence", "0%")
                    try: return int(str(c).replace("%", "").strip())
                    except: return 0

                all_defects = [d for d in all_defects if _conf_val(d) >= conf_threshold]

                # Deduplication is handled upstream by unified_processor._nms

                # Group defects for UI display
                # Note: From the taxonomy, Group I is "Fabric Structure" (Hole, Tear, Missing Thread, Slub, Snag, Oil Stain).
                # Wait, "Surface" vs "Structural" was the old taxonomy.
                # Let's cleanly split them based on the Types. 
                # Structural: Hole, Tear, Missing Thread, Snag, plus all Stitch defects.
                # Surface: Slub, Oil Stain.
                struc_types = {"Hole", "Tear", "Snag", "Missing Thread", "Skip Stitch", "Broken Stitch", "Run-off Stitch", "Crooked Stitch", "Pucker"}
                structural = [d for d in all_defects if d.get("Type") in struc_types]
                surface = [d for d in all_defects if d not in structural]

                s_count = len(structural)
                f_count = len(surface)
                total = s_count + f_count
                verdict = "PASS" if total == 0 else "FAIL"
                badge = '<span class="badge-pass">✓ PASS</span>' if total == 0 else '<span class="badge-fail">✗ FAIL</span>'

                # ── ROUTING INFO ──
                with st.expander("🧠 Routing Details", expanded=False):
                    pre_info = routing_info.get("pre_classification", {})
                    st.markdown(f"""
                    **Pre-classification:** Seam detected: `{pre_info.get('has_seam', False)}` · Fabric body: `{pre_info.get('has_fabric_body', True)}`  
                    **Engines used:** {', '.join(routing_info.get('engines_used', []))}  
                    **Defect types found:** {', '.join(result['summary'].get('defect_types_found', [])) or 'None'}
                    """)

                # ── METRICS ──
                st.markdown(f"""
                <div class="metrics-row">
                    <div class="m-card"><div class="m-val">{badge}</div><div class="m-lbl">Verdict</div></div>
                    <div class="m-card"><div class="m-val">{total}</div><div class="m-lbl">Total Defects</div></div>
                    <div class="m-card"><div class="m-val" style="-webkit-text-fill-color: var(--structural-clr); color: var(--structural-clr);">{s_count}</div><div class="m-lbl">Structural</div></div>
                    <div class="m-card"><div class="m-val" style="-webkit-text-fill-color: var(--surface-clr); color: var(--surface-clr);">{f_count}</div><div class="m-lbl">Surface</div></div>
                </div>
                """, unsafe_allow_html=True)

                # ── VISUAL RESULTS: Original · Structural · Surface ──
                st.markdown('<div class="sec-title">Visual Results</div>', unsafe_allow_html=True)

                base_img = decode_image(img_file)

                structural_overlay = draw_category_overlay(base_img, structural, (113, 113, 248), "S")  # red-ish BGR
                surface_overlay = draw_category_overlay(base_img, surface, (36, 191, 251), "F")  # yellow BGR

                c1, c2, c3 = st.columns(3)
                with c1:
                    with st.container(border=True):
                        st.caption("📷  Original")
                        st.image(base_img, channels="BGR", use_container_width=True)
                with c2:
                    with st.container(border=True):
                        st.caption(f"🔴  Structural Defects ({s_count})")
                        st.image(structural_overlay, channels="BGR", use_container_width=True)
                with c3:
                    with st.container(border=True):
                        st.caption(f"🟡  Surface Defects ({f_count})")
                        st.image(surface_overlay, channels="BGR", use_container_width=True)

                # ── BEFORE / AFTER COMPARISON ──
                if total > 0:
                    st.markdown('<div class="sec-title">Before / After Comparison</div>', unsafe_allow_html=True)
                    # Merge both overlays into one annotated image
                    annotated = draw_category_overlay(base_img, structural, (113, 113, 248), "S")
                    annotated = draw_category_overlay(annotated, surface, (36, 191, 251), "F")
                    with st.container(border=True):
                        st.caption("Drag the slider to compare original vs annotated")
                        render_comparison_slider(base_img, annotated)

                # ── DIAGNOSTIC MAPS ──
                st.markdown('<div class="sec-title">Diagnostic Maps</div>', unsafe_allow_html=True)
                map_cols = st.columns(min(3, max(1, len(viz_maps))))
                map_items = list(viz_maps.items())
                map_labels = {
                    "entropy": "🧵 Entropy Heatmap (Texture)",
                    "saliency": "📡 Saliency Map (Spectral+DWT)",
                    "anomaly_heatmap": "📐 Anomaly Heatmap (Edge+Frangi)",
                    "seam_output": "🪡 Seam Detection Output",
                    "ssim_heatmap": "🖼️ SSIM Deviation Map (Reference)",
                    "reference_result": "🔍 Reference Compare Result",
                }
                for idx, (key, viz_img) in enumerate(map_items):
                    col_idx = idx % len(map_cols)
                    with map_cols[col_idx]:
                        with st.container(border=True):
                            label = map_labels.get(key, key.replace("_", " ").title())
                            st.caption(label)
                            if viz_img is not None:
                                # Apply colormap if grayscale
                                if len(viz_img.shape) == 2:
                                    viz_img = cv2.applyColorMap(viz_img, cv2.COLORMAP_JET)
                                st.image(viz_img, channels="BGR", use_container_width=True)


                # ── DEFECT TABLES ──
                st.markdown('<div class="sec-title">Defect Report</div>', unsafe_allow_html=True)

                if total == 0:
                    with st.container(border=True):
                        st.success("✅  All inspections passed — no defects detected.")
                else:
                    if structural:
                        st.markdown('<div class="cat-hdr cat-structural">🔴  Structural Defects — Physical Damage</div>', unsafe_allow_html=True)
                        df_s = pd.DataFrame(structural)
                        preferred = ["Category", "Group", "Engine", "Inspector", "Type", "Area (px)", "Solidity", "Confidence", "Location"]
                        cols_order = [c for c in preferred if c in df_s.columns] + [c for c in df_s.columns if c not in preferred and not c.startswith("bbox")]
                        with st.container(border=True):
                            st.dataframe(df_s[cols_order], use_container_width=True)

                    if surface:
                        st.markdown('<div class="cat-hdr cat-surface">🟡  Surface Defects — Visual / Textural</div>', unsafe_allow_html=True)
                        df_f = pd.DataFrame(surface)
                        preferred = ["Category", "Group", "Engine", "Inspector", "Type", "Area (px)", "Solidity", "Confidence", "Location"]
                        cols_order = [c for c in preferred if c in df_f.columns] + [c for c in df_f.columns if c not in preferred and not c.startswith("bbox")]
                        with st.container(border=True):
                            st.dataframe(df_f[cols_order], use_container_width=True)

                    # Combined download
                    df_all = pd.DataFrame(all_defects)
                    preferred = ["Category", "Group", "Engine", "Inspector", "Type", "Area (px)", "Solidity", "Confidence", "Location"]
                    cols_order = [c for c in preferred if c in df_all.columns] + [c for c in df_all.columns if c not in preferred and not c.startswith("bbox")]
                    csv = df_all[cols_order].to_csv(index=False).encode("utf-8")
                    st.download_button("⬇  Download Full Report (CSV)", csv, "fabricqa_report.csv", mime="text/csv")

                    # PDF Report
                    try:
                        pdf_bytes = generate_pdf_report(base_img, structural_overlay, surface_overlay, all_defects, verdict, s_count, f_count)
                        st.download_button("📄  Download PDF Report", pdf_bytes, "fabricqa_report.pdf", mime="application/pdf")
                    except Exception:
                        pass  # silently skip if PDF fails

                # ── USER FEEDBACK ──
                if total > 0:
                    st.markdown('<div class="sec-title">Feedback</div>', unsafe_allow_html=True)
                    st.caption("Help improve detection — mark any false positives")
                    for i, d in enumerate(all_defects[:20]):
                        col_desc, col_btn = st.columns([4, 1])
                        with col_desc:
                            st.text(f"{d.get('Type', 'Unknown')} | {d.get('Confidence', '')} | {d.get('Inspector', '')}")
                        with col_btn:
                            if st.button("❌ False Positive", key=f"fb_{i}"):
                                save_feedback({
                                    "timestamp": datetime.now().isoformat(),
                                    "filename": getattr(img_file, 'name', 'unknown'),
                                    "defect_type": d.get("Type", ""),
                                    "confidence": d.get("Confidence", ""),
                                    "inspector": d.get("Inspector", ""),
                                    "action": "false_positive"
                                })
                                st.toast("✓ Feedback saved", icon="✅")

                # Save to history
                save_history({
                    "timestamp": datetime.now().isoformat(),
                    "filename": getattr(img_file, 'name', 'camera_capture'),
                    "defect_count": total,
                    "structural": s_count,
                    "surface": f_count,
                    "verdict": verdict,
                })

            except Exception as exc:
                st.error(f"Inspection failed: {exc}")
                with st.expander("Debug traceback"):
                    st.exception(exc)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📤</div>
            <p>Upload a fabric image to get a complete quality report.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sec-title">Batch Input</div>', unsafe_allow_html=True)

    with st.container(border=True):
        batch_files = st.file_uploader(
            "Upload multiple fabric images",
            type=["jpg", "png", "bmp"],
            accept_multiple_files=True,
            key="batch_upload",
        )
        batch_sens = st.slider("Sensitivity", 1.0, 5.0, 2.5, 0.1, key="batch_sens")
        batch_shadows = st.checkbox("Remove Shadows", value=False, key="batch_shadows")

    if batch_files:
        st.markdown('<div class="sec-title">Batch Results</div>', unsafe_allow_html=True)
        summary_data = []
        all_batch_defects = []
        progress = st.progress(0, text="Starting batch…")

        for idx, bf in enumerate(batch_files):
            progress.progress(idx / len(batch_files), text=f"🔍 Inspecting {bf.name}  ({idx+1}/{len(batch_files)})")

            _, err = validate_image(bf)
            if err:
                summary_data.append({"Filename": bf.name, "Structural": "—", "Surface": "—", "Total": "ERR", "Verdict": err})
                continue

            try:
                # ── UNIFIED PROCESSOR (Batch) ──
                buf = clone_buffer(bf)
                result = unified_processor.process(buf, sensitivity=batch_sens, mode="full", remove_shadows=batch_shadows)
                defects = result["defects"]

                for d in defects:
                    d["Category"] = classify_defect(d.get("Type", ""))
                    d["Filename"] = bf.name

                s_count = sum(1 for d in defects if d["Category"] == "Structural")
                f_count = sum(1 for d in defects if d["Category"] == "Surface")
                total = len(defects)
                verdict = "PASS" if total == 0 else "FAIL"

                summary_data.append({
                    "Filename": bf.name,
                    "Structural": s_count, "Surface": f_count,
                    "Total": total, "Verdict": verdict,
                })
                all_batch_defects.extend(defects)

                save_history({
                    "timestamp": datetime.now().isoformat(),
                    "filename": bf.name,
                    "defect_count": total, "structural": s_count,
                    "surface": f_count, "verdict": verdict,
                })
            except Exception as exc:
                summary_data.append({"Filename": bf.name, "Structural": "—", "Surface": "—", "Total": "ERR", "Verdict": str(exc)})

        progress.progress(1.0, text="✅ Batch complete!")

        df_s = pd.DataFrame(summary_data)
        passes = len(df_s[df_s["Verdict"] == "PASS"])
        fails = len(df_s[df_s["Verdict"] == "FAIL"])

        st.markdown(f"""
        <div class="metrics-row">
            <div class="m-card"><div class="m-val">{len(batch_files)}</div><div class="m-lbl">Processed</div></div>
            <div class="m-card"><div class="m-val">{passes}</div><div class="m-lbl">Passed</div></div>
            <div class="m-card"><div class="m-val">{fails}</div><div class="m-lbl">Failed</div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.dataframe(df_s, use_container_width=True)

        if all_batch_defects:
            df_all = pd.DataFrame(all_batch_defects)
            preferred = ["Filename", "Category", "Type", "Confidence"]
            cols_order = [c for c in preferred if c in df_all.columns] + [c for c in df_all.columns if c not in preferred]
            csv = df_all[cols_order].to_csv(index=False).encode("utf-8")
            st.download_button("⬇  Download Batch Report", csv, "batch_report.csv", mime="text/csv")

            # ── DEFECT CLUSTER HEATMAP ──
            heatmap_img = generate_defect_heatmap(all_batch_defects)
            if heatmap_img is not None:
                st.markdown('<div class="sec-title">🔥 Defect Cluster Heatmap</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.caption("Aggregate view — brighter regions have more defects across all images")
                    st.image(heatmap_img, channels="BGR", use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="sec-title">Inspection History</div>', unsafe_allow_html=True)
    history = load_history()

    if not history:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📊</div>
            <p>No inspection history yet. Run an inspection first.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(history)
        total = len(df_hist)
        passes = len(df_hist[df_hist["verdict"] == "PASS"])
        fails = total - passes
        rate = f'{passes/total*100:.0f}%' if total > 0 else '—'

        st.markdown(f"""
        <div class="metrics-row">
            <div class="m-card"><div class="m-val">{total}</div><div class="m-lbl">Inspections</div></div>
            <div class="m-card"><div class="m-val">{passes}</div><div class="m-lbl">Passed</div></div>
            <div class="m-card"><div class="m-val">{fails}</div><div class="m-lbl">Failed</div></div>
            <div class="m-card"><div class="m-val">{rate}</div><div class="m-lbl">Pass Rate</div></div>
        </div>
        """, unsafe_allow_html=True)

        if "defect_count" in df_hist.columns:
            st.markdown('<div class="sec-title">Defect Trend</div>', unsafe_allow_html=True)
            chart_data = df_hist[["defect_count"]].rename(columns={"defect_count": "Defects"})
            st.bar_chart(chart_data)

        st.markdown('<div class="sec-title">Recent Inspections</div>', unsafe_allow_html=True)
        display_cols = [c for c in ["timestamp", "filename", "defect_count", "structural", "surface", "verdict"] if c in df_hist.columns]
        with st.container(border=True):
            st.dataframe(df_hist[display_cols].iloc[::-1], use_container_width=True)

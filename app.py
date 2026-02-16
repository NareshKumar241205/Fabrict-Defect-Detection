"""
FabricQA — Automated Fabric Defect Detection
==============================================
Upload an image → all inspectors run → results categorized
into Structural (red) and Surface (yellow) defects.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from io import BytesIO
from datetime import datetime

from inspectors import texture_inspector, spectral_inspector, seam_inspector, edge_inspector
from config import SEAM_SETTINGS

# ──────────────────────────────────────────────
# DEFECT CLASSIFICATION MAP
# ──────────────────────────────────────────────
STRUCTURAL_TYPES = {
    "Cut / Tear (Horiz)", "Cut / Tear (Vert)",
    "Horizontal Tear/Thread", "Vertical Tear/Thread",
    "Ragged Hole", "Skip Stitch",
    "Structural Break", "Weave Irregularity",
}
SURFACE_TYPES = {
    "Oil / Water Stain", "Texture Defect", "Rough Weave",
    "Texture Anomaly", "Wrinkle / Fold (Horiz)", "Wrinkle / Fold (Vert)",
    "Deviation",
}

def classify_defect(defect_type: str) -> str:
    """Categorize a defect as Structural or Surface."""
    if defect_type in STRUCTURAL_TYPES:
        return "Structural"
    if defect_type in SURFACE_TYPES:
        return "Surface"
    # fallback heuristic
    low = defect_type.lower()
    if any(k in low for k in ("tear", "hole", "cut", "stitch", "break", "weave")):
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


def decode_image(img_file):
    """Read an image file into a BGR numpy array."""
    img_file.seek(0)
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img_file.seek(0)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


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

    # ── PROCESS ──
    if img_file is not None:
        orig_img, validation_error = validate_image(img_file)
        if validation_error:
            st.warning(validation_error)
        elif orig_img is not None:
            try:
                all_defects = []
                progress = st.progress(0, text="⏳ Starting inspection…")

                # 1. Texture
                progress.progress(0.0, text="🧵  Analyzing texture (LBP + Gabor)…")
                buf = clone_buffer(img_file)
                _, _, ent_map, _, tex_defs = texture_inspector.detect_defects(buf, sensitivity=sensitivity)
                for d in tex_defs: d["Inspector"] = "Texture"
                all_defects.extend(tex_defs)

                # 2. Spectral
                progress.progress(0.25, text="📡  Spectral analysis (Multi-Res FFT)…")
                buf = clone_buffer(img_file)
                _, _, sal_map, spec_defs = spectral_inspector.detect_defects(buf, sensitivity=sensitivity)
                for d in spec_defs: d["Inspector"] = "Spectral"
                all_defects.extend(spec_defs)

                # 3. Seam
                progress.progress(0.50, text="🪡  Seam / stitch detection…")
                buf = clone_buffer(img_file)
                _, _, _, _, seam_defs = seam_inspector.detect_defects(buf)
                for d in seam_defs: d["Inspector"] = "Seam"
                all_defects.extend(seam_defs)

                # 4. Edge
                progress.progress(0.75, text="📐  Edge / structure analysis…")
                buf = clone_buffer(img_file)
                _, _, anomaly_heatmap, _, edge_defs = edge_inspector.detect_defects(buf, sensitivity=sensitivity)
                for d in edge_defs: d["Inspector"] = "Edge"
                all_defects.extend(edge_defs)

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

                structural = [d for d in all_defects if d["Category"] == "Structural"]
                surface = [d for d in all_defects if d["Category"] == "Surface"]

                s_count = len(structural)
                f_count = len(surface)
                total = s_count + f_count
                verdict = "PASS" if total == 0 else "FAIL"
                badge = '<span class="badge-pass">✓ PASS</span>' if total == 0 else '<span class="badge-fail">✗ FAIL</span>'

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

                # ── DIAGNOSTIC MAPS ──
                st.markdown('<div class="sec-title">Diagnostic Maps</div>', unsafe_allow_html=True)
                mc1, mc2 = st.columns(2)
                with mc1:
                    with st.container(border=True):
                        st.caption("Entropy Heatmap (Texture)")
                        st.image(cv2.applyColorMap(ent_map, cv2.COLORMAP_JET), channels="BGR", use_container_width=True)
                with mc2:
                    with st.container(border=True):
                        st.caption("Saliency Map (Spectral)")
                        st.image(sal_map, channels="BGR", use_container_width=True)

                # ── DEFECT TABLES ──
                st.markdown('<div class="sec-title">Defect Report</div>', unsafe_allow_html=True)

                if total == 0:
                    with st.container(border=True):
                        st.success("✅  All inspections passed — no defects detected.")
                else:
                    if structural:
                        st.markdown('<div class="cat-hdr cat-structural">🔴  Structural Defects — Physical Damage</div>', unsafe_allow_html=True)
                        df_s = pd.DataFrame(structural)
                        preferred = ["Category", "Inspector", "Type", "Area (px)", "Solidity", "Confidence", "Location"]
                        cols_order = [c for c in preferred if c in df_s.columns] + [c for c in df_s.columns if c not in preferred and not c.startswith("bbox")]
                        with st.container(border=True):
                            st.dataframe(df_s[cols_order], use_container_width=True)

                    if surface:
                        st.markdown('<div class="cat-hdr cat-surface">🟡  Surface Defects — Visual / Textural</div>', unsafe_allow_html=True)
                        df_f = pd.DataFrame(surface)
                        preferred = ["Category", "Inspector", "Type", "Area (px)", "Solidity", "Confidence", "Location"]
                        cols_order = [c for c in preferred if c in df_f.columns] + [c for c in df_f.columns if c not in preferred and not c.startswith("bbox")]
                        with st.container(border=True):
                            st.dataframe(df_f[cols_order], use_container_width=True)

                    # Combined download
                    df_all = pd.DataFrame(all_defects)
                    preferred = ["Category", "Inspector", "Type", "Area (px)", "Solidity", "Confidence", "Location"]
                    cols_order = [c for c in preferred if c in df_all.columns] + [c for c in df_all.columns if c not in preferred and not c.startswith("bbox")]
                    csv = df_all[cols_order].to_csv(index=False).encode("utf-8")
                    st.download_button("⬇  Download Full Report (CSV)", csv, "fabricqa_report.csv", mime="text/csv")

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
                defects = []
                # Texture
                buf = clone_buffer(bf)
                _, _, _, _, defs = texture_inspector.detect_defects(buf, sensitivity=batch_sens)
                defects.extend(defs)
                # Spectral
                buf = clone_buffer(bf)
                _, _, _, defs = spectral_inspector.detect_defects(buf, sensitivity=batch_sens)
                defects.extend(defs)
                # Seam
                buf = clone_buffer(bf)
                _, _, _, _, defs = seam_inspector.detect_defects(buf)
                defects.extend(defs)
                # Edge
                buf = clone_buffer(bf)
                _, _, _, _, defs = edge_inspector.detect_defects(buf, sensitivity=batch_sens)
                defects.extend(defs)

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

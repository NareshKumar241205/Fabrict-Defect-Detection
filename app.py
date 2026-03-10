# app.py
import os

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from fpdf import FPDF
import tempfile
import base64

from inspectors.unified_processor import unified_processor

# STRICT TAXONOMY RESTRICTION
ALLOWED_TYPES = {
    # Pipeline 1 (N)
    "Oil Stain", "Hole", "Tear",
    # Pipeline 2 (A)
    "Slub", "Skip/Miss Stitch", "Crooked Stitch", "Skip Stitch", "Broken Stitch", "Run-off Stitch"
}

STRUCTURAL_TYPES = {
    "Hole", "Tear", "Skip/Miss Stitch", "Crooked Stitch", "Skip Stitch", "Broken Stitch", "Run-off Stitch"
}
SURFACE_TYPES = {
    "Oil Stain", "Slub"
}

DEFECT_COLORS = {
    "Hole": (60, 60, 255), "Tear": (50, 50, 220), "Skip/Miss Stitch": (80, 80, 240),
    "Crooked Stitch": (200, 0, 200), "Skip Stitch": (150, 0, 150), "Broken Stitch": (100, 0, 100), "Run-off Stitch": (120, 100, 200),
    "Slub": (30, 180, 255), "Oil Stain": (20, 140, 220),
}

import json
from datetime import datetime

def save_history(entry):
    try:
        with open("inspection_history.json", "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(entry)
    with open("inspection_history.json", "w") as f:
        json.dump(history, f, indent=2)

def load_history():
    try:
        with open("inspection_history.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

st.set_page_config(page_title="Parallel Pipeline FabricQA", layout="wide")

def clone_buffer(f):
    if f is None: return None
    f.seek(0)
    return BytesIO(f.read())

def decode_image(img_file):
    img_file.seek(0)
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def validate_image(img_file):
    try:
        img = decode_image(img_file)
        if img is None:
            return None, "Invalid image file or unsupported format."
        return img, None
    except Exception as e:
        return None, f"Error loading image: {str(e)}"

def classify_defect(defect_type):
    if defect_type in STRUCTURAL_TYPES:
        return "Structural"
    elif defect_type in SURFACE_TYPES:
        return "Surface"
    else:
        return "Unknown"

def deduplicate_defects(defects, iou_threshold=0.5):
    """Remove overlapping defects based on IoU threshold."""
    if not defects:
        return defects
    
    # Sort by quality score (highest first) to keep better detections
    defects = sorted(defects, key=lambda d: _conf_val(d), reverse=True)
    
    kept = []
    for defect in defects:
        # Check if this defect overlaps significantly with any kept defect
        should_keep = True
        for kept_defect in kept:
            if calculate_iou(defect, kept_defect) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(defect)
    
    return kept

def calculate_iou(defect1, defect2):
    """Calculate Intersection over Union for two defects."""
    x1_1 = defect1.get("bbox_x", 0)
    y1_1 = defect1.get("bbox_y", 0)
    w1 = defect1.get("bbox_w", 0)
    h1 = defect1.get("bbox_h", 0)
    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1
    
    x1_2 = defect2.get("bbox_x", 0)
    y1_2 = defect2.get("bbox_y", 0)
    w2 = defect2.get("bbox_w", 0)
    h2 = defect2.get("bbox_h", 0)
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def _conf_val(d):
    c = d.get("Quality Score", "0%")
    try: return int(str(c).replace("%", "").strip())
    except: return 0


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


def generate_pdf_report(base_img, structural_overlay, surface_overlay, all_defects, verdict, s_count, f_count):
    """Generate a PDF report with inspection results."""
    try:
        pdf = FPDF()
        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 12, "Parallel Pipeline FabricQA - Inspection Report", new_x="LMARGIN", new_y="NEXT")
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

        # Save images temporarily
        tmp_files = []
        for label, img in [("Original", base_img), ("Structural", structural_overlay), ("Surface", surface_overlay)]:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                tmp_files.append(tmp.name)

        # Images
        img_w = 58
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

            headers = ["#", "Category", "Type", "Area (px)", "Quality Score (%)"]
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
                pdf.cell(col_w[4], 6, str(d.get("Quality Score", "")), border=1)
                pdf.ln()

        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_bytes = pdf_output.getvalue()

        # Clean up temp files
        for f in tmp_files:
            try:
                os.unlink(f)
            except:
                pass

        return pdf_bytes
    except Exception as e:
        raise e


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
        conf = d.get("Quality Score", "")
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
    <h1>Parallel Pipeline FabricQA</h1>
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
            st.markdown("**⚙️ Detection Parameters**")
            
            # Detection Threshold Multiplier with info
            with st.expander("ℹ️ Detection Threshold Multiplier", expanded=False):
                st.markdown("""
                **Mathematical Concept**: Controls the trade-off between False Positives (FP) and True Positives (TP).
                
                **Formula**: `threshold = mean_diff + (4.5 - multiplier × 0.6) × std_diff`
                
                - **Higher values** (4.0-5.0): More conservative, fewer false positives but may miss defects
                - **Lower values** (1.0-2.0): More aggressive, catches more defects but increases false alarms
                - **Default** (2.5): Balanced detection sensitivity
                """)
            
            detection_threshold = st.slider("Detection Threshold Multiplier", 1.0, 5.0, 2.5, 0.1,
                                          help="Controls false positive vs true positive trade-off",
                                          key="sens_slider")
            
            # Quality Score Threshold with info
            with st.expander("ℹ️ Quality Score Threshold", expanded=False):
                st.markdown("""
                **Mathematical Concept**: Minimum reliability score for defect acceptance.
                
                **Calculation**: Based on geometric properties like solidity and shape consistency.
                
                **Formula**: `score = 50 + (solidity × 40)` (Classic Edge) or custom metrics (Logic inspectors)
                
                - **Higher threshold** (60-80%): Only high-confidence defects shown
                - **Lower threshold** (0-30%): All potential defects displayed
                - **Default** (30%): Balanced quality filtering
                """)
            
            quality_threshold = st.slider("Quality Score Threshold (%)", 0, 80, 30, 5,
                                        help="Minimum reliability score for defect acceptance",
                                        key="conf_slider")
            
            remove_shadows_ui = st.checkbox("Remove Shadows (Illumination Correction)", value=False,
                                          help="Enable this if uneven lighting is causing false positive defects",
                                          key="shadow_toggle")

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
                result = unified_processor.process(buf, sensitivity=detection_threshold, mode="full", remove_shadows=remove_shadows_ui)
                
                progress.progress(0.9, text="📊  Compiling results…")
                all_defects = result["defects"]
                viz_maps = result["viz_maps"]
                routing_info = result["routing_info"]
                
                progress.progress(1.0, text="✅  Inspection complete!")

                # ── CLASSIFY & FILTER ──
                for d in all_defects:
                    d["Category"] = classify_defect(d.get("Type", ""))

                # Parse quality score string "42%" → 42, filter below threshold
                def _conf_val(d):
                    c = d.get("Quality Score", "0%")
                    try: return int(str(c).replace("%", "").strip())
                    except: return 0

                all_defects = [d for d in all_defects if _conf_val(d) >= quality_threshold]

                # Deduplicate overlapping boxes across inspectors
                all_defects = deduplicate_defects(all_defects, iou_threshold=0.5)

                structural = [d for d in all_defects if d["Category"] == "Structural"]
                surface = [d for d in all_defects if d["Category"] == "Surface"]

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
                        st.image(base_img, channels="BGR", width="stretch")
                with c2:
                    with st.container(border=True):
                        st.caption(f"🔴  Structural Defects ({s_count})")
                        st.image(structural_overlay, channels="BGR", width="stretch")
                with c3:
                    with st.container(border=True):
                        st.caption(f"🟡  Surface Defects ({f_count})")
                        st.image(surface_overlay, channels="BGR", width="stretch")

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
                    "classical_edge_map": "🔍 Statistical Background Subtraction (Sauvola)",
                    "logic_spectral_map": "📡 Multi-Scale FFT Spectral Residual",
                    "logic_seam_map": "🪡 Hough-Aligned Thread Projection",
                    "dwt_map": "🧵 DWT (db4) Texture Decomposition",
                    "grayscale_clahe": "🖼️ Grayscale + CLAHE Output",
                    "shadow_removal_mask": "🌑 Shadow Removal Mask",
                    "projection_profile": "📈 1D Projection Profile",
                    "binary_thread_mask": "🧵 Binary Thread Mask",
                    "deskewed_image": "🔄 Deskewed Image",
                    "multi_scale_saliency_512": "📡 FFT Saliency (512)",
                    "multi_scale_saliency_256": "📡 FFT Saliency (256)",
                    "multi_scale_saliency_128": "📡 FFT Saliency (128)",
                    "dwt_map": "🧵 DWT (db4) Texture Decomposition",
                    "final_binary_mask": "🔲 Final Binary Mask",
                    "adaptive_threshold_map": "⚖️ Adaptive Threshold Map",
                }
                for idx, (key, viz_img) in enumerate(map_items):
                    col_idx = idx % len(map_cols)
                    with map_cols[col_idx]:
                        with st.container(border=True):
                            label = map_labels.get(key, key.replace("_", " ").title())
                            st.caption(label)
                            if viz_img is not None:
                                if isinstance(viz_img, np.ndarray):
                                    if len(viz_img.shape) == 2:
                                        # Apply colormap if grayscale
                                        viz_img = cv2.applyColorMap(viz_img, cv2.COLORMAP_JET)
                                        st.image(viz_img, channels="BGR", width="stretch")
                                    elif len(viz_img.shape) == 1:
                                        # Plot 1D array as line chart
                                        st.line_chart(viz_img)
                                    else:
                                        st.write(f"Unsupported array shape: {viz_img.shape}")
                                else:
                                    st.write(f"Unsupported viz type: {type(viz_img)}")


                # ── DEFECT TABLES ──
                st.markdown('<div class="sec-title">Defect Report</div>', unsafe_allow_html=True)

                if total == 0:
                    with st.container(border=True):
                        st.success("✅  All inspections passed — no defects detected.")
                else:
                    if structural:
                        st.markdown('<div class="cat-hdr cat-structural">🔴  Structural Defects — Physical Damage</div>', unsafe_allow_html=True)
                        df_s = pd.DataFrame(structural)
                        preferred = ["Category", "Group", "Engine", "Inspector", "Type", "Area (px)", "Solidity", "Quality Score", "Location"]
                        cols_order = [c for c in preferred if c in df_s.columns] + [c for c in df_s.columns if c not in preferred and not c.startswith("bbox")]
                        with st.container(border=True):
                            st.dataframe(df_s[cols_order], width="stretch")

                    if surface:
                        st.markdown('<div class="cat-hdr cat-surface">🟡  Surface Defects — Visual / Textural</div>', unsafe_allow_html=True)
                        df_f = pd.DataFrame(surface)
                        preferred = ["Category", "Group", "Engine", "Inspector", "Type", "Area (px)", "Solidity", "Quality Score", "Location"]
                        cols_order = [c for c in preferred if c in df_f.columns] + [c for c in df_f.columns if c not in preferred and not c.startswith("bbox")]
                        with st.container(border=True):
                            st.dataframe(df_f[cols_order], width="stretch")

                    # Combined download
                    df_all = pd.DataFrame(all_defects)
                    preferred = ["Category", "Group", "Engine", "Inspector", "Type", "Area (px)", "Solidity", "Quality Score", "Location"]
                    cols_order = [c for c in preferred if c in df_all.columns] + [c for c in df_all.columns if c not in preferred and not c.startswith("bbox")]
                    csv = df_all[cols_order].to_csv(index=False).encode("utf-8")
                    st.download_button("⬇  Download Full Report (CSV)", csv, "parallel_pipeline_fabricqa_report.csv", mime="text/csv")

                    # PDF Report
                    try:
                        pdf_bytes = generate_pdf_report(base_img, structural_overlay, surface_overlay, all_defects, verdict, s_count, f_count)
                        st.download_button("📄  Download PDF Report", pdf_bytes, "parallel_pipeline_fabricqa_report.pdf", mime="application/pdf")
                    except Exception:
                        pass  # silently skip if PDF fails

                # ── USER FEEDBACK ──
                if total > 0:
                    st.markdown('<div class="sec-title">Feedback</div>', unsafe_allow_html=True)
                    st.caption("Help improve detection — mark any false positives")
                    for i, d in enumerate(all_defects[:20]):
                        col_desc, col_btn = st.columns([4, 1])
                        with col_desc:
                            st.text(f"{d.get('Type', 'Unknown')} | {d.get('Quality Score', '')} | {d.get('Inspector', '')}")
                        with col_btn:
                            if st.button("❌ False Positive", key=f"fb_{i}"):
                                save_feedback({
                                    "timestamp": datetime.now().isoformat(),
                                    "filename": getattr(img_file, 'name', 'unknown'),
                                    "defect_type": d.get("Type", ""),
                                    "quality_score": d.get("Quality Score", ""),
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
        batch_detection_threshold = st.slider("Detection Threshold Multiplier", 1.0, 5.0, 2.5, 0.1, key="batch_sens")
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
                result = unified_processor.process(buf, sensitivity=batch_detection_threshold, mode="full", remove_shadows=batch_shadows)
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
            st.dataframe(df_s, width="stretch")

        if all_batch_defects:
            df_all = pd.DataFrame(all_batch_defects)
            preferred = ["Filename", "Category", "Type", "Quality Score"]
            cols_order = [c for c in preferred if c in df_all.columns] + [c for c in df_all.columns if c not in preferred]
            csv = df_all[cols_order].to_csv(index=False).encode("utf-8")
            st.download_button("⬇  Download Batch Report", csv, "batch_report.csv", mime="text/csv")

            # ── DEFECT CLUSTER HEATMAP ──
            heatmap_img = generate_defect_heatmap(all_batch_defects)
            if heatmap_img is not None:
                st.markdown('<div class="sec-title">🔥 Defect Cluster Heatmap</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.caption("Aggregate view — brighter regions have more defects across all images")
                    st.image(heatmap_img, channels="BGR", width="stretch")


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
            st.dataframe(df_hist[display_cols].iloc[::-1], width="stretch")

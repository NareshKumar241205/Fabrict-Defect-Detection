import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import json
import os
import io
from datetime import datetime
from inspectors import inspector, spectral_inspector, stitch_inspector, edge_detector, glcm_inspector

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Fabric Inspector Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Dark Premium Theme) ---
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
    }
    
    /* Status Banners */
    .status-pass {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #ecfdf5;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        border: 1px solid #10b981;
    }
    .status-fail {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fef2f2;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        border: 1px solid #ef4444;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    
    /* Engine Badge */
    .engine-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .engine-spectral { background: #312e81; color: #a5b4fc; }
    .engine-lbp { background: #064e3b; color: #6ee7b7; }
    .engine-seam { background: #78350f; color: #fcd34d; }
    .engine-edge { background: #7f1d1d; color: #fca5a5; }
    .engine-glcm { background: #4c1d95; color: #c4b5fd; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    st.subheader("📥 Input Source")
    input_src = st.radio("Select Source", ["Upload Image", "Live Camera"], label_visibility="collapsed")
    
    st.divider()
    
    with st.expander("🎛️ Engine Calibration", expanded=False):
        st.caption("**Surface Engines**")
        spectral_sensitivity = st.slider("Spectral (Structural)", 1.0, 5.0, 3.0, 0.1,
                                         help="FFT-based detection. Lower = more sensitive.")
        lbp_sensitivity = st.slider("LBP (Texture/Stains)", 1.0, 5.0, 3.0, 0.1,
                                     help="Entropy-based detection for surface stains.")
        
        st.caption("**Structural Engines**")
        glcm_contrast = st.slider("GLCM Contrast Threshold", 100, 500, 300, 10,
                                   help="Higher = less false positives for weave defects.")
        glcm_correlation = st.slider("GLCM Correlation Threshold", 0.1, 0.9, 0.4, 0.05,
                                      help="Lower = stricter classification.")
        
        st.caption("**Linear Engines**")
        edge_min_length = st.slider("Min Line Length (Edge)", 50, 300, 150, 10,
                                     help="Only detect lines longer than this.")
        edge_gradient_thresh = st.slider("Gradient Threshold", 50, 300, 150, 10)
        
        st.caption("**Seam Engine**")
        laplacian_thresh = st.slider("Seam Pucker Threshold", 100, 400, 200, 10,
                                      help="Laplacian variance threshold for pucker detection.")
    
    st.divider()
    
    with st.expander("🔧 Display Options", expanded=False):
        show_debug = st.checkbox("Show Debug Maps", value=False)
        show_overlay = st.checkbox("Show Heatmap Overlay", value=False)
        overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.4, 0.05) if show_overlay else 0.4

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🔬 Fabric Inspector Pro</h1>
    <p>Automated Optical Inspection System — 5 Detection Engines</p>
</div>
""", unsafe_allow_html=True)

# --- INPUT ---
img_file = None
if input_src == "Upload Image":
    img_file = st.file_uploader("Feed Material", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
else:
    img_file = st.camera_input("Scan Material")

# --- PROCESSING PIPELINE ---
if img_file is not None:
    start_time = time.time()
    
    all_defects = []
    debug_images = {}
    engine_results = {}
    
    # Load image once
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    orig_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    progress = st.progress(0, text="Initializing engines...")
    
    try:
        # --- Engine 1: Spectral Residual (Structural) ---
        progress.progress(40, text="🔬 Engine 1/2 — Spectral Residual Analysis...")
        img_file.seek(0)
        _, spectral_result, saliency_heatmap, spectral_defects = spectral_inspector.detect_defects(
            img_file, sensitivity=spectral_sensitivity
        )
        all_defects.extend([{**d, "Source": "Spectral"} for d in spectral_defects])
        debug_images["Saliency Map"] = saliency_heatmap
        engine_results["spectral"] = spectral_result
        
        # --- Engine 2: LBP Texture (Stains/Surface) ---
        progress.progress(80, text="🧵 Engine 2/2 — LBP Texture Analysis...")
        img_file.seek(0)
        _, _, entropy_map, lbp_result, lbp_defects = inspector.detect_defects(
            img_file, lbp_sensitivity, min_area=100
        )
        all_defects.extend([{**d, "Source": "LBP"} for d in lbp_defects])
        # Fix: Colorize entropy map to prevent Streamlit error (must be 3 channels for channels="BGR")
        debug_images["Entropy Map"] = cv2.applyColorMap(entropy_map, cv2.COLORMAP_JET)
        engine_results["lbp"] = lbp_result
        
        # REMOVED: Stitch, Edge, and GLCM engines (per user request to restore original logic)
        progress.progress(100, text="Analysis Complete!")

    except ValueError as e:
        progress.empty()
        st.error(f"Error analyzing image: {str(e)}")
        st.stop()
    except Exception as e:
        progress.empty()
        st.error(f"Unexpected error: {str(e)}")
        st.stop()
    
    processing_time = (time.time() - start_time) * 1000
    defect_count = len(all_defects)
    progress.progress(100, text="✅ Analysis Complete!")
    time.sleep(0.3)
    progress.empty()
    
    # --- METRICS ROW ---
    engine_counts = {}
    for d in all_defects:
        src = d.get("Source", "Unknown")
        engine_counts[src] = engine_counts.get(src, 0) + 1
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{defect_count}</div>
            <div class="label">Total Defects</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{processing_time:.0f}ms</div>
            <div class="label">Cycle Time</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="value">5</div>
            <div class="label">Engines Run</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{engine_counts.get('Spectral', 0) + engine_counts.get('LBP', 0)}</div>
            <div class="label">Surface Defects</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{engine_counts.get('Edge', 0) + engine_counts.get('GLCM', 0)}</div>
            <div class="label">Structural Defects</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- STATUS BANNER ---
    if defect_count == 0:
        st.markdown('<div class="status-pass">✅ PASS — Material Certified OK</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-fail">❌ FAIL — {defect_count} Defect(s) Detected</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- TABBED RESULTS ---
    col_main, col_manifest = st.columns([2, 1])
    
    with col_main:
        tab_surface, tab_structural = st.tabs([
            "🔬 Surface (LBP)", "🏗️ Structural (Spectral)"
        ])
        
        with tab_surface:
            if lbp_defects:
                st.image(engine_results["lbp"], caption="LBP / Texture Defects", channels="BGR", use_container_width=True)
            else:
                st.success("No surface texture defects detected.")
        
        with tab_structural:
            if show_overlay and saliency_heatmap is not None:
                # Resize heatmap to match result image
                heatmap_resized = cv2.resize(saliency_heatmap, (spectral_result.shape[1], spectral_result.shape[0]))
                overlay = cv2.addWeighted(spectral_result, 1 - overlay_alpha, heatmap_resized, overlay_alpha, 0)
                st.image(overlay, caption="Spectral Analysis + Saliency Overlay", channels="BGR", use_container_width=True)
            else:
                st.image(spectral_result, caption="Spectral Analysis (FFT Residual)", channels="BGR", use_container_width=True)
    
    with col_manifest:
        st.subheader("📋 Defect Manifest")
        if defect_count > 0:
            df = pd.DataFrame(all_defects)
            # Select common columns that exist across all engines
            display_cols = ["Source", "Type"]
            if "Location" in df.columns:
                display_cols.append("Location")
            display_df = df[display_cols]
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("No anomalies detected.")
        
        # --- EXPORT BUTTONS ---
        st.markdown("---")
        st.subheader("📥 Export")
        
        # CSV Export
        if defect_count > 0:
            csv_data = pd.DataFrame(all_defects).to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name=f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Image Export
        _, img_encoded = cv2.imencode('.jpg', spectral_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        st.download_button(
            label="🖼️ Download Annotated Image",
            data=img_encoded.tobytes(),
            file_name=f"inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    
    # --- DEBUG MAPS ---
    if show_debug:
        with st.expander("🔧 Technician View — Debug Maps", expanded=True):
            debug_cols = st.columns(len(debug_images))
            for i, (k, v) in enumerate(debug_images.items()):
                debug_cols[i].image(v, caption=k, channels="BGR", use_container_width=True)
    
    # --- SAVE TO HISTORY ---
    history_file = os.path.join(os.path.dirname(__file__), "inspection_log.json")
    record = {
        "timestamp": datetime.now().isoformat(),
        "filename": img_file.name if hasattr(img_file, 'name') else "camera_capture",
        "defect_count": defect_count,
        "verdict": "PASS" if defect_count == 0 else "FAIL",
        "processing_time_ms": round(processing_time, 1),
        "defects_by_engine": engine_counts,
        "defects": all_defects
    }
    
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []
    
    history.append(record)
    
    # Keep only last 1000 records
    if len(history) > 1000:
        history = history[-1000:]
        
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #1e1e2e, #2d2d44); 
                border-radius: 12px; border: 1px dashed rgba(255,255,255,0.15); margin-top: 1rem;">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">🔬</p>
        <p style="color: #e2e8f0; font-size: 1.2rem; font-weight: 500;">Feed the machine to start inspection</p>
        <p style="color: #94a3b8; font-size: 0.85rem;">Upload an image or use the camera to begin automated analysis</p>
    </div>
    """, unsafe_allow_html=True)
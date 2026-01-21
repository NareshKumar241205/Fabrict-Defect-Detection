import streamlit as st
import cv2
import numpy as np
import pandas as pd
from processor import inspector

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Fabric Inspector Pro",
    layout="wide",
    page_icon="🧵",
    initial_sidebar_state="expanded"
)

# --- THEME DEFINITIONS ---
THEMES = {
    "Dark Mode": {
        "bg": "#0E1117", "card": "#262730", "text": "#FAFAFA",
        "accent": "#00E5FF", "sidebar": "#262730", "border": "#00E5FF",
        "success_bg": "#1B5E20", "success_txt": "#A5D6A7",
        "fail_bg": "#B71C1C", "fail_txt": "#FFCDD2"
    },
    "Light Mode": {
        "bg": "#F8F9FA", "card": "#FFFFFF", "text": "#212529",
        "accent": "#0068C9", "sidebar": "#FFFFFF", "border": "#DEE2E6",
        "success_bg": "#D4EDDA", "success_txt": "#155724",
        "fail_bg": "#F8D7DA", "fail_txt": "#721C24"
    }
}

# --- SIDEBAR & THEME SELECTION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fabrics.png", width=60)
    st.title("Control Panel")
    
    # Theme Toggle
    st.markdown("### 🎨 Theme")
    selected_theme_name = st.radio("UI Mode", ["Dark Mode", "Light Mode"], horizontal=True)
    t = THEMES[selected_theme_name] # Current Theme Data

    st.divider()
    
    # Inputs
    st.subheader("📡 Input Source")
    input_src = st.radio("Select Feed", ["Upload Batch", "Live Camera"])
    
    st.divider()
    
    # Settings
    st.subheader("🎛️ Settings")
    sensitivity = st.slider("Strictness (Sigma)", 1.0, 5.0, 2.5, 0.1)
    min_area = st.slider("Min Area (px)", 50, 1000, 250)
    
    st.success("System: **ONLINE**")

# --- CSS INJECTION ---
st.markdown(f"""
<style>
    .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
    section[data-testid="stSidebar"] {{ background-color: {t['sidebar']}; }}
    
    /* Metrics */
    div[data-testid="stMetricValue"] {{ color: {t['accent']}; font-size: 2rem; }}
    div[data-testid="stMetricLabel"] {{ color: {t['text']}; opacity: 0.8; }}
    .stMetric {{
        background-color: {t['card']};
        border: 1px solid {t['border']};
        border-left: 5px solid {t['accent']};
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    /* Status Boxes */
    .status-box {{
        padding: 20px; border-radius: 8px; text-align: center;
        font-weight: bold; font-size: 1.5rem; margin-bottom: 20px;
    }}
    .pass {{ background-color: {t['success_bg']}; color: {t['success_txt']}; border: 2px solid {t['success_txt']}; }}
    .fail {{ background-color: {t['fail_bg']}; color: {t['fail_txt']}; border: 2px solid {t['fail_txt']}; }}
    
    h1, h2, h3, p {{ color: {t['text']}; }}
</style>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🏭 Fabric Inspector Pro")
    st.caption("Automated Optical Inspection (AOI) | LBP-u2 Engine")

img_file = None
if input_src == "Upload Batch":
    img_file = st.file_uploader("Upload Sample", type=['jpg', 'png'])
else:
    img_file = st.camera_input("Conveyor Feed")

if img_file:
    # PROCESS
    _, _, _, result, data = inspector.detect_defects(img_file, sensitivity, min_area)
    defect_count = len(data)
    quality = max(0, 100 - (defect_count * 15))

    # STATUS BANNER
    if defect_count == 0:
        st.markdown(f'<div class="status-box pass">✅ QA PASSED: CLEAN FABRIC</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-box fail">🚫 QA FAILED: {defect_count} DEFECTS FOUND</div>', unsafe_allow_html=True)

    # VISUALS & DATA
    col_vis, col_data = st.columns([2, 1])
    
    with col_vis:
        st.markdown("### 👁️ Visual Feed")
        # Fixed for Streamlit 2025+ compliance
        st.image(result, channels="BGR", width="stretch")

    with col_data:
        st.markdown("### 📊 Telemetry")
        m1, m2 = st.columns(2)
        m1.metric("Defects", f"{defect_count}")
        m2.metric("Quality", f"{quality}%")
        
        st.markdown("### 📝 Log")
        if defect_count > 0:
            df = pd.DataFrame(data)
            st.dataframe(
                df[["Type", "Area (px)"]], 
                hide_index=True, 
                use_container_width=True
            )
        else:
            st.info("No anomalies detected.")

else:
    st.warning("⚠️ Waiting for Input Signal...")
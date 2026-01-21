import streamlit as st
import cv2
import numpy as np
import pandas as pd
from processor import inspector
from config import DEFAULT_GLCM, DEFAULT_SEAM, DEFAULT_SYSTEM

st.set_page_config(
    page_title="AOI System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 24px; }
        h1, h2, h3 { font-family: 'Helvetica', sans-serif; font-weight: 600; }
        .stButton>button { width: 100%; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Control Panel")
    
    st.subheader("Input Source")
    input_src = st.radio(
        "Select Source",
        ["Upload Image", "Live Camera"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.subheader("Inspection Mode")
    target_mode = st.radio(
        "Select Target",
        ["Fabric Surface", "Stitch / Seam"]
    )

    st.markdown("---")
    
    with st.expander("Advanced Calibration", expanded=True):
        st.info("Adjust thresholds for specific fabric types/colors.")
        
        # 1. GLOBAL
        st.markdown("**Global Parameters**")
        bg_thresh = st.slider("Background Threshold", 0, 255, DEFAULT_SYSTEM["BACKGROUND_THRESH"])

        settings_override = {"BACKGROUND_THRESH": bg_thresh}

        # 2. FABRIC
        if target_mode == "Fabric Surface":
            st.markdown("**Texture Parameters**")
            cont_max = st.slider("Contrast Max (Slub/Hole)", 100, 500, DEFAULT_GLCM["CONTRAST_MAX"])
            corr_min = st.slider("Correlation Min (Structure)", 0.50, 1.00, DEFAULT_GLCM["CORRELATION_MIN"])
            homo_max = st.slider("Homogeneity Max (Stains)", 0.80, 1.00, DEFAULT_GLCM["HOMOGENEITY_MAX"])

            settings_override.update({
                "CONTRAST_MAX": cont_max,
                "CORRELATION_MIN": corr_min,
                "HOMOGENEITY_MAX": homo_max
            })

        # 3. SEAM
        elif target_mode == "Stitch / Seam":
            st.markdown("**Geometry Parameters**")
            stitch_bright = st.slider("Thread Brightness", 50, 255, DEFAULT_SEAM["STITCH_COLOR_THRESH"])
            gap_tol = st.slider("Max Gap Tolerance (px)", 1, 50, DEFAULT_SEAM["GAP_TOLERANCE"])

            settings_override.update({
                "STITCH_COLOR_THRESH": stitch_bright,
                "GAP_TOLERANCE": gap_tol
            })

st.title("Automated Optical Inspection")
st.markdown("Fabric Defect Detection & Quality Assurance System")
st.markdown("---")

# IMAGE INPUT
img_file = None
if input_src == "Upload Image":
    img_file = st.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"])
else:
    img_file = st.camera_input("Capture Image")

# PROCESSING
if img_file is not None:
    mode_key = "fabric" if target_mode == "Fabric Surface" else "seam"
    
    # Process
    result_img, defect_data = inspector.process(
        img_file, 
        mode=mode_key, 
        settings=settings_override
    )

    defect_count = len(defect_data)
    
    # STATUS INDICATOR (Plain text)
    if defect_count == 0:
        st.success("PASS: Material meets quality standards.")
    else:
        st.error(f"FAIL: {defect_count} anomalies detected.")

    # DISPLAY RESULTS
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Visual Inspection")
        st.image(result_img, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("Statistics")
        st.metric("Total Defects", defect_count)
        
        if defect_count > 0:
            df = pd.DataFrame(defect_data)
            st.dataframe(df, hide_index=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Export Report", csv, "report.csv", "text/csv")

else:
    st.info("Please provide an image input to begin inspection.")

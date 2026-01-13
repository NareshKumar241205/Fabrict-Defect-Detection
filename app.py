import streamlit as st
import cv2
import numpy as np
import pandas as pd
from processor import process_image, auto_calibrate

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fabric Inspector", layout="wide")
st.title("Fabric Defect Inspection System")

# --- SESSION STATE (Memory) ---
# We use this to remember the calibration settings even if the page reloads
if 'tex_sens' not in st.session_state:
    st.session_state['tex_sens'] = 180 # Default Factory Setting
if 'dark_thresh' not in st.session_state:
    st.session_state['dark_thresh'] = 90 # Default Factory Setting
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("Controls")
input_mode = st.sidebar.radio("Input Source", ["Upload Image", "Live Camera"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("1. System Calibration")
st.sidebar.info("Upload a clean fabric sample to auto-tune the system.")

calib_image = st.sidebar.file_uploader("Upload 'Good' Sample", type=["jpg", "png", "jpeg"])

if calib_image is not None:
    if st.sidebar.button("Run Auto-Calibration"):
        with st.spinner("Analyzing fabric texture statistics..."):
            rec_sens, rec_dark = auto_calibrate(calib_image)
            # Update Session State
            st.session_state['tex_sens'] = rec_sens
            st.session_state['dark_thresh'] = rec_dark
            st.session_state['is_calibrated'] = True
        st.sidebar.success(f"Calibrated! Sensitivity: {rec_sens}, Dark Level: {rec_dark}")

# Status Indicator
if st.session_state['is_calibrated']:
    st.sidebar.markdown(f"**Status:** ✅ Optimized for current fabric.")
else:
    st.sidebar.markdown(f"**Status:** ⚠️ Using Standard Defaults.")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Quality Standards")

# Simple Strictness Control (Adjusts the calibrated values slightly)
sens_level = st.sidebar.select_slider("Detection Strictness", options=["Low", "Medium", "High"], value="Medium")

# Apply Strictness Logic
final_sensitivity = st.session_state['tex_sens']
final_dark = st.session_state['dark_thresh']

if sens_level == "Low":
    final_sensitivity -= 20 # Relaxed
    final_dark -= 20
elif sens_level == "High":
    final_sensitivity += 20 # Strict
    final_dark += 20

# Noise Filter (User can override)
min_area = st.sidebar.slider("Ignore Defects Smaller Than (px)", 100, 5000, 500, help="Defects smaller than this are considered negligible noise.")

# --- MAIN INSPECTION AREA ---
st.markdown("### 🕵️ Live Inspection")

image_buffer = None
if input_mode == "Upload Image":
    image_buffer = st.file_uploader("Upload Test Image (Defective)", type=["jpg", "png", "jpeg"])
elif input_mode == "Live Camera":
    image_buffer = st.camera_input("Take a snapshot")

if image_buffer is not None:
    # RUN PROCESSING using the calculated values
    original, mask, result, data = process_image(image_buffer, final_sensitivity, min_area, final_dark)

    # METRICS DISPLAY
    col1, col2, col3 = st.columns(3)
    
    # PASS / FAIL LOGIC
    if len(data) == 0:
        st.success("✅ **STATUS: PASS** - Fabric is Error Free")
        col1.metric("Status", "PASS", delta_color="normal")
    else:
        st.error(f"❌ **STATUS: FAIL** - Found {len(data)} Defects")
        col1.metric("Status", "FAIL", delta_color="inverse")
    
    col2.metric("Defects Found", len(data))
    
    # RESULT TABS
    tab1, tab2, tab3 = st.tabs(["🔍 Final Result", "⚫ Debug Mask", "📄 Report Data"])
    with tab1:
        st.image(result, channels="BGR", use_container_width=True)
    with tab2:
        st.image(mask, use_container_width=True)
    with tab3:
        if len(data) > 0:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "report.csv", "text/csv")
        else:
            st.info("No defects to report.")

else:
    st.info("Waiting for input...")
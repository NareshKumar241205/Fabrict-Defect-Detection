import streamlit as st
import cv2
import numpy as np
import pandas as pd
from processor import inspector

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fabric Inspector",
    layout="wide",
    page_icon="🧵"
)

# --- SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Input Selection
    input_src = st.radio("Input Source", ["Upload Image", "Live Camera"])
    
    st.divider()
    
    # Tuning Parameters
    st.subheader("Detection Sensitivity")
    sensitivity = st.slider("Strictness Level", 1.0, 5.0, 2.5, 0.1)
    min_area = st.slider("Min Defect Size (px)", 50, 500, 200)
    
    st.info("Ready to scan.")

# --- MAIN PAGE ---
st.title("🧵 Fabric Quality Inspection")
st.markdown("Automated defect detection system.")

# --- LOGIC ---
img_file = None
if input_src == "Upload Image":
    img_file = st.file_uploader("Upload a fabric image", type=['jpg', 'png', 'jpeg'])
else:
    img_file = st.camera_input("Take a picture")

if img_file:
    # 1. Run Analysis
    _, _, _, result_img, defect_data = inspector.detect_defects(img_file, sensitivity, min_area)
    
    defect_count = len(defect_data)
    quality_score = max(0, 100 - (defect_count * 15))

    # 2. Display Status (Pass/Fail)
    if defect_count == 0:
        st.success("✅ **PASS**: No defects detected.")
    else:
        st.error(f"🚫 **FAIL**: Found {defect_count} anomalies.")

    st.divider()

    # 3. Layout: Image on Left, Data on Right
    col_image, col_stats = st.columns([2, 1])

    with col_image:
        st.subheader("Visual Result")
        st.image(result_img, channels="BGR", use_container_width=True)

    with col_stats:
        st.subheader("Analysis Data")
        
        # Simple Metrics
        m1, m2 = st.columns(2)
        m1.metric("Defect Count", defect_count)
        m2.metric("Quality Score", f"{quality_score}%")
        
        # Data Table
        if defect_count > 0:
            st.warning("Defect Log:")
            df = pd.DataFrame(defect_data)
            # Show specific columns only
            st.dataframe(df[["Type", "Area (px)"]], use_container_width=True, hide_index=True)
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report",
                csv,
                "defect_report.csv",
                "text/csv"
            )
        else:
            st.markdown("*System is running normally.*")

else:
    # Idle State
    st.info("👋 Please upload an image or use the camera to start.")
# app.py
import streamlit as st
import pandas as pd
import time
from processor import engine

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Fabric Inspector Pro",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Product" Feel
st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .stMetric {
        background-color: #262730;
        border: 1px solid #4B4B4B;
        border-radius: 5px;
        padding: 10px;
    }
    .status-pass {
        color: #00C853; font-weight: bold; font-size: 1.2em; border: 1px solid #00C853;
        padding: 10px; border-radius: 5px; text-align: center;
    }
    .status-fail {
        color: #D50000; font-weight: bold; font-size: 1.2em; border: 1px solid #D50000;
        padding: 10px; border-radius: 5px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.markdown("### 1. Inspection Mode")
    mode = st.radio(
        "Target Logic",
        ["Fabric Surface", "Seam Assembly"],
        help="Surface: Checks texture (GLCM). Seam: Checks geometry."
    )
    
    st.divider()
    
    st.markdown("### 2. Input Source")
    src = st.radio("Feed Type", ["Upload Image", "Live Camera"])
    
    st.divider()
    st.caption(f"System Status: ONLINE")
    st.caption(f"Active Engine: {mode}")

# --- MAIN DASHBOARD ---
st.title("🏭 Fabric Quality Assurance System")
st.markdown(f"**Current Task:** Automated Optical Inspection (AOI) for *{mode}*.")

# --- INPUT HANDLING ---
img_file = None
if src == "Upload Image":
    img_file = st.file_uploader("Upload Batch Sample", type=['jpg', 'png', 'jpeg'])
else:
    img_file = st.camera_input("Conveyor Line Feed")

# --- EXECUTION ---
if img_file:
    try:
        start_time = time.time()
        
        # CALL THE ENGINE
        result_img, defects = engine.run_inspection(img_file, mode)
        
        # Calculate Latency
        process_time = (time.time() - start_time) * 1000
        
        # --- LAYOUT ---
        col_vis, col_data = st.columns([2, 1])
        
        with col_vis:
            st.subheader("👁️ Visual Analysis")
            st.image(result_img, channels="BGR", use_container_width=True)
            
        with col_data:
            st.subheader("📊 Telemetry")
            
            # KPI Metrics
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Defects Found", len(defects))
            kpi2.metric("Latency", f"{process_time:.0f} ms", delta_color="inverse")
            
            st.divider()
            
            # Status Banner
            if len(defects) == 0:
                st.markdown('<div class="status-pass">✅ QA PASSED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-fail">🚫 QA FAILED</div>', unsafe_allow_html=True)
                
            # Defect Log
            if defects:
                st.markdown("#### Defect Log")
                df = pd.DataFrame(defects)
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.info("No structural or geometric anomalies detected.")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 Waiting for input stream...")
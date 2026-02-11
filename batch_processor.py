"""
Batch Processor Module
======================
Allows processing multiple fabric images at once and generates
a summary dashboard with pass/fail stats and exportable reports.

Usage: streamlit run batch_processor.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import io
from datetime import datetime
from inspectors import inspector, spectral_inspector, stitch_inspector, edge_detector, glcm_inspector

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Batch Fabric Inspector",
    page_icon="📦",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    
    .batch-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .batch-header h1 { color: #fff; font-weight: 700; font-size: 1.8rem; margin: 0; }
    .batch-header p { color: #a0aec0; font-size: 0.9rem; margin: 0.3rem 0 0 0; }
    
    .summary-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .summary-card .value { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
    .summary-card .label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .pass-value { color: #34d399 !important; }
    .fail-value { color: #f87171 !important; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="batch-header">
    <h1>📦 Batch Fabric Inspector</h1>
    <p>Upload multiple images for bulk inspection with summary analytics</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Batch Settings")
    
    with st.expander("🎛️ Engine Calibration", expanded=False):
        spectral_sensitivity = st.slider("Spectral Sensitivity", 1.0, 5.0, 3.0, 0.1)
        lbp_sensitivity = st.slider("LBP Sensitivity", 1.0, 5.0, 3.0, 0.1)
        laplacian_thresh = st.slider("Seam Threshold", 100, 400, 200, 10)
        edge_min_length = st.slider("Min Line Length", 50, 300, 150, 10)
        edge_gradient_thresh = st.slider("Gradient Threshold", 50, 300, 150, 10)
        glcm_contrast = st.slider("GLCM Contrast", 100, 500, 300, 10)
        glcm_correlation = st.slider("GLCM Correlation", 0.1, 0.9, 0.4, 0.05)

# --- UPLOAD ---
uploaded_files = st.file_uploader(
    "Upload Fabric Images for Batch Inspection",
    type=['jpg', 'png', 'jpeg'],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    st.markdown("---")
    
    all_results = []
    all_defects_global = []
    
    overall_progress = st.progress(0, text="Starting batch processing...")
    
    for idx, img_file in enumerate(uploaded_files):
        progress_pct = int((idx / len(uploaded_files)) * 100)
        overall_progress.progress(progress_pct, text=f"Processing {idx + 1}/{len(uploaded_files)}: {img_file.name}")
        
        start_time = time.time()
        all_defects = []
        
        try:
            # Engine 1: Spectral
            img_file.seek(0)
            _, _, _, spectral_defects = spectral_inspector.detect_defects(img_file, sensitivity=spectral_sensitivity)
            all_defects.extend([{**d, "Source": "Spectral"} for d in spectral_defects])
            
            # Engine 2: LBP
            img_file.seek(0)
            _, _, _, _, lbp_defects = inspector.detect_defects(img_file, lbp_sensitivity, min_area=100)
            all_defects.extend([{**d, "Source": "LBP"} for d in lbp_defects])
            
            # Engine 3: Seam
            img_file.seek(0)
            _, _, seam_defects = stitch_inspector.check_seam_quality(img_file, laplacian_threshold=laplacian_thresh)
            all_defects.extend([{**d, "Source": "Seam"} for d in seam_defects])
            
            # Engine 4: Edge
            img_file.seek(0)
            _, _, _, edge_defects = edge_detector.detect_linear_defects(
                img_file, min_line_length=edge_min_length, gradient_threshold=edge_gradient_thresh
            )
            all_defects.extend([{**d, "Source": "Edge"} for d in edge_defects])
            
            # Engine 5: GLCM
            img_file.seek(0)
            _, _, _, glcm_defects = glcm_inspector.detect_structural_defects(
                img_file, contrast_threshold=glcm_contrast, correlation_threshold=glcm_correlation
            )
            all_defects.extend([{**d, "Source": "GLCM"} for d in glcm_defects])
            
            processing_time = (time.time() - start_time) * 1000
            defect_count = len(all_defects)
            
            result = {
                "Filename": img_file.name,
                "Verdict": "PASS" if defect_count == 0 else "FAIL",
                "Defects": defect_count,
                "Spectral": sum(1 for d in all_defects if d["Source"] == "Spectral"),
                "LBP": sum(1 for d in all_defects if d["Source"] == "LBP"),
                "Seam": sum(1 for d in all_defects if d["Source"] == "Seam"),
                "Edge": sum(1 for d in all_defects if d["Source"] == "Edge"),
                "GLCM": sum(1 for d in all_defects if d["Source"] == "GLCM"),
                "Time (ms)": round(processing_time, 1)
            }
        except Exception as e:
            result = {
                "Filename": img_file.name,
                "Verdict": "ERROR",
                "Defects": 0,
                "Spectral": 0, "LBP": 0, "Seam": 0, "Edge": 0, "GLCM": 0,
                "Time (ms)": 0
            }
            # Add a dummy defect to explain the error
            all_defects.append({
                "Filename": img_file.name,
                "Source": "System",
                "Type": "Processing Error",
                "Location": "N/A",
                "Details": str(e)
            })
        all_results.append(result)
        
        for d in all_defects:
            d["Filename"] = img_file.name
        all_defects_global.extend(all_defects)
    
    overall_progress.progress(100, text="✅ Batch processing complete!")
    time.sleep(0.3)
    overall_progress.empty()
    
    # --- SUMMARY DASHBOARD ---
    results_df = pd.DataFrame(all_results)
    total = len(results_df)
    passed = len(results_df[results_df["Verdict"] == "PASS"])
    failed = total - passed
    total_defects = results_df["Defects"].sum()
    avg_time = results_df["Time (ms)"].mean()
    
    st.markdown("## 📊 Summary Dashboard")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="summary-card"><div class="value">{total}</div><div class="label">Total Images</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="summary-card"><div class="value pass-value">{passed}</div><div class="label">Passed</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="summary-card"><div class="value fail-value">{failed}</div><div class="label">Failed</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="summary-card"><div class="value">{total_defects}</div><div class="label">Total Defects</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="summary-card"><div class="value">{avg_time:.0f}ms</div><div class="label">Avg. Cycle Time</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- CHARTS ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Defects by Engine")
        engine_totals = {
            "Spectral": results_df["Spectral"].sum(),
            "LBP": results_df["LBP"].sum(),
            "Seam": results_df["Seam"].sum(),
            "Edge": results_df["Edge"].sum(),
            "GLCM": results_df["GLCM"].sum()
        }
        engine_df = pd.DataFrame({"Engine": engine_totals.keys(), "Defects": engine_totals.values()})
        st.bar_chart(engine_df.set_index("Engine"))
    
    with chart_col2:
        st.subheader("Pass/Fail Distribution")
        verdict_df = pd.DataFrame({"Verdict": ["PASS", "FAIL"], "Count": [passed, failed]})
        st.bar_chart(verdict_df.set_index("Verdict"))
    
    # --- RESULTS TABLE ---
    st.markdown("---")
    st.subheader("📋 Detailed Results")
    st.dataframe(results_df, hide_index=True, use_container_width=True)
    
    # --- PER-IMAGE DETAILS ---
    with st.expander("🔍 Per-Image Defect Details"):
        for result in all_results:
            fname = result["Filename"]
            defects_for_file = [d for d in all_defects_global if d.get("Filename") == fname]
            if defects_for_file:
                st.markdown(f"**{fname}** — {len(defects_for_file)} defect(s)")
                file_df = pd.DataFrame(defects_for_file)
                display_cols = [c for c in ["Source", "Type", "Location"] if c in file_df.columns]
                st.dataframe(file_df[display_cols], hide_index=True, use_container_width=True)
            else:
                st.markdown(f"**{fname}** — ✅ No defects")
    
    # --- BATCH EXPORT ---
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv_summary = results_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Summary CSV",
            data=csv_summary,
            file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        if all_defects_global:
            csv_details = pd.DataFrame(all_defects_global).to_csv(index=False)
            st.download_button(
                label="📄 Download All Defects CSV",
                data=csv_details,
                file_name=f"batch_defects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #1e1e2e, #2d2d44); 
                border-radius: 12px; border: 1px dashed rgba(255,255,255,0.15); margin-top: 1rem;">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">📦</p>
        <p style="color: #e2e8f0; font-size: 1.2rem; font-weight: 500;">Upload multiple images for batch inspection</p>
        <p style="color: #94a3b8; font-size: 0.85rem;">Select multiple files at once for bulk quality analysis</p>
    </div>
    """, unsafe_allow_html=True)

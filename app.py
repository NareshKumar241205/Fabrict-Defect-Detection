import streamlit as st
import cv2
import numpy as np
import pandas as pd
from processor import inspector


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fabric Quality Inspector",
    page_icon="🧵",
    layout="wide"
)


# =====================================================
# SIDEBAR – INSPECTION CONTROLS
# =====================================================
with st.sidebar:
    st.markdown("## ⚙️ Inspection Settings")

    # -----------------------------
    # INPUT SOURCE
    # -----------------------------
    st.markdown("### 📥 Input Source")
    input_src = st.radio(
        "Select input type",
        ["Upload Image", "Live Camera"],
        help="Choose whether to inspect a saved fabric image or capture one live."
    )

    st.divider()

    # -----------------------------
    # INSPECTION MODE
    # -----------------------------
    st.markdown("### 🎯 Inspection Mode")

    mode = st.radio(
        "Mode",
        ["🟢 Normal", "🟡 Strict", "🔴 Ultra-Strict"],
        help=(
            "🟢 Normal → General fabric inspection\n"
            "🟡 Strict → Detect subtle texture defects\n"
            "🔴 Ultra-Strict → Detect seams, missing threads & stitch issues"
        )
    )

    # Mode → sensitivity mapping
    if mode == "🟢 Normal":
        sensitivity = 2.0
        st.info("Balanced detection, minimal false alarms.")
    elif mode == "🟡 Strict":
        sensitivity = 3.0
        st.warning("Higher sensitivity. May detect subtle anomalies.")
    else:
        sensitivity = 4.5
        st.error("Maximum sensitivity. Seam & stitch focused.")

    st.caption(f"🔧 Internal sensitivity = **{sensitivity}**")

    st.divider()

    # -----------------------------
    # DEFECT FILTERING
    # -----------------------------
    st.markdown("### 🧹 Defect Filtering")

    min_area = st.slider(
        "Minimum Defect Size (px)",
        min_value=50,
        max_value=500,
        value=200,
        help="Ignore very small noise-like regions."
    )

    st.caption("⬆ Larger value → fewer false positives")

    st.divider()

    # -----------------------------
    # SYSTEM STATUS
    # -----------------------------
    st.markdown("### 🟢 System Status")
    st.success("Ready to scan")


# =====================================================
# MAIN DASHBOARD
# =====================================================
st.markdown(
    """
    # 🧵 Fabric Quality Inspection System  
    **Classical Computer Vision–based Automated Optical Inspection (AOI)**  
    *(No Machine Learning / No Deep Learning)*
    """
)

st.divider()


# =====================================================
# IMAGE INPUT
# =====================================================
img_file = None
if input_src == "Upload Image":
    img_file = st.file_uploader(
        "📂 Upload a fabric image",
        type=["jpg", "jpeg", "png"]
    )
else:
    img_file = st.camera_input("📸 Capture fabric image")


# =====================================================
# PROCESSING & RESULTS
# =====================================================
if img_file is not None:
    _, _, _, result_img, defect_data = inspector.detect_defects(
        img_file,
        sensitivity=sensitivity,
        min_area=min_area
    )

    defect_count = len(defect_data)
    quality_score = max(0, 100 - defect_count * 15)

    # -----------------------------
    # PASS / FAIL BANNER
    # -----------------------------
    if defect_count == 0:
        st.success("✅ PASS — No defects detected.")
    else:
        st.error(f"🚫 FAIL — {defect_count} defect(s) detected.")

    st.divider()

    # -----------------------------
    # RESULTS LAYOUT
    # -----------------------------
    col_img, col_stats = st.columns([2.2, 1])

    with col_img:
        st.markdown("## 🖼 Visual Inspection Result")
        st.image(
            result_img,
            channels="BGR",
            use_container_width=True
        )

    with col_stats:
        st.markdown("## 📊 Inspection Summary")

        m1, m2 = st.columns(2)
        m1.metric("Defect Count", defect_count)
        m2.metric("Quality Score", f"{quality_score}%")

        if defect_count > 0:
            st.markdown("### 🧾 Defect Log")

            df = pd.DataFrame(defect_data)
            st.dataframe(
                df[["Type", "Area (px)"]],
                use_container_width=True,
                hide_index=True
            )

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Inspection Report",
                data=csv,
                file_name="fabric_defect_report.csv",
                mime="text/csv"
            )
        else:
            st.markdown(
                """
                🟢 **Fabric condition is normal**  
                No structural or texture anomalies detected.
                """
            )

else:
    st.info("👈 Upload an image or use the camera to begin inspection.")

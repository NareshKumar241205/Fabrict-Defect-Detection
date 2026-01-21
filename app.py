import streamlit as st
import pandas as pd
from processor import engine

# --- CONFIG ---
st.set_page_config(page_title="Fabric AI", layout="wide", page_icon="🧵")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .status-pass { padding: 15px; background: #064E3B; border: 1px solid #34D399; border-radius: 8px; text-align: center; }
    .status-fail { padding: 15px; background: #7F1D1D; border: 1px solid #F87171; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Inspection Mode")
    
    # CRITICAL: This selection changes the math used
    mode = st.radio(
        "Select Material / Task",
        ["Textured / Denim", "Smooth Fabric", "Seam Assembly"]
    )
    
    st.divider()
    
    if mode == "Textured / Denim":
        st.info("ℹ️ **Engine:** DoG (Difference of Gaussians)\n\nIgnores heavy weaves. Detects structural breaks.")
    elif mode == "Smooth Fabric":
        st.info("ℹ️ **Engine:** LBP + Entropy\n\nDetects subtle surface variations.")
    else:
        st.info("ℹ️ **Engine:** 1D Signal Processing\n\nDetects rhythm breaks in stitching.")

# --- MAIN ---
st.title(f"🏭 Defect Detection: {mode}")

img_file = st.file_uploader("Upload Sample", type=['jpg', 'png'])

if img_file:
    # 1. PROCESS
    result, log, debug = engine.run(img_file, mode)
    
    # 2. STATUS BANNER
    if not log:
        st.markdown('<div class="status-pass">✅ QA PASSED: NO DEFECTS</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-fail">🚫 QA FAILED: {len(log)} DEFECTS FOUND</div>', unsafe_allow_html=True)
    
    st.divider()

    # 3. VISUALIZATION
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.image(result, channels="BGR", caption="Final Inspection Result", use_container_width=True)
        
        # Show "X-Ray" view for debugging
        if debug is not None:
            with st.expander("Show Computer Vision Filter (Debug)"):
                st.image(debug, caption="Algorithm View (Noise Filtered)", clamp=True, use_container_width=True)
    
    with c2:
        if log:
            st.subheader("Defect Log")
            df = pd.DataFrame(log)
            # Clean up dataframe for display
            display_df = df[["Type", "Area"]]
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.success("Material is within tolerance levels.")

else:
    st.info("Waiting for input stream...")
"""
Inspection History Viewer
=========================
Displays past inspection results stored in inspection_log.json.

Usage: streamlit run history.py
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Inspection History",
    page_icon="📜",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    
    .history-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .history-header h1 { color: #fff; font-weight: 700; font-size: 1.8rem; margin: 0; }
    .history-header p { color: #a0aec0; font-size: 0.9rem; margin: 0.3rem 0 0 0; }
    
    .stat-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stat-card .value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; }
    .stat-card .label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="history-header">
    <h1>📜 Inspection History</h1>
    <p>Review past inspection results and quality trends</p>
</div>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
history_file = os.path.join(os.path.dirname(__file__), "inspection_log.json")

if not os.path.exists(history_file):
    st.info("No inspection history found. Run an inspection in the main app first.")
    st.stop()

try:
    with open(history_file, 'r') as f:
        history = json.load(f)
except (json.JSONDecodeError, IOError):
    st.error("Error reading inspection log. The file may be corrupted.")
    st.stop()

if not history:
    st.info("No inspection records yet.")
    st.stop()

# --- PARSE HISTORY ---
rows = []
for record in history:
    rows.append({
        "Timestamp": record.get("timestamp", "Unknown"),
        "Filename": record.get("filename", "Unknown"),
        "Verdict": record.get("verdict", "Unknown"),
        "Defects": record.get("defect_count", 0),
        "Time (ms)": record.get("processing_time_ms", 0),
        "Spectral": record.get("defects_by_engine", {}).get("Spectral", 0),
        "LBP": record.get("defects_by_engine", {}).get("LBP", 0),
        "Seam": record.get("defects_by_engine", {}).get("Seam", 0),
        "Edge": record.get("defects_by_engine", {}).get("Edge", 0),
        "GLCM": record.get("defects_by_engine", {}).get("GLCM", 0),
    })

df = pd.DataFrame(rows)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

# --- STATS ---
total = len(df)
passed = len(df[df["Verdict"] == "PASS"])
failed = total - passed
pass_rate = (passed / total * 100) if total > 0 else 0
avg_time = df["Time (ms)"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="stat-card"><div class="value">{total}</div><div class="label">Total Inspections</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="value" style="color:#34d399">{passed}</div><div class="label">Passed</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="value" style="color:#f87171">{failed}</div><div class="label">Failed</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card"><div class="value">{pass_rate:.1f}%</div><div class="label">Pass Rate</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="stat-card"><div class="value">{avg_time:.0f}ms</div><div class="label">Avg. Cycle Time</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- FILTERS ---
with st.sidebar:
    st.title("🔍 Filters")
    
    verdict_filter = st.multiselect("Verdict", ["PASS", "FAIL"], default=["PASS", "FAIL"])
    
    if len(df) > 1:
        min_date = df["Timestamp"].min().date()
        max_date = df["Timestamp"].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date))
    else:
        date_range = None

# Apply filters
filtered_df = df[df["Verdict"].isin(verdict_filter)]
if date_range and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["Timestamp"].dt.date >= start_date) &
        (filtered_df["Timestamp"].dt.date <= end_date)
    ]

# --- TRENDS ---
st.subheader("📈 Quality Trend")
if len(filtered_df) > 1:
    trend_df = filtered_df[["Timestamp", "Defects"]].copy()
    trend_df = trend_df.set_index("Timestamp")
    st.line_chart(trend_df)
else:
    st.info("Need at least 2 inspections for trend analysis.")

# --- HISTORY TABLE ---
st.markdown("---")
st.subheader("📋 Inspection Log")

display_df = filtered_df.copy()
display_df["Timestamp"] = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
st.dataframe(display_df, hide_index=True, use_container_width=True)

# --- EXPORT ---
st.markdown("---")
col_e1, col_e2 = st.columns(2)

with col_e1:
    csv_data = display_df.to_csv(index=False)
    st.download_button(
        label="📄 Download History CSV",
        data=csv_data,
        file_name=f"inspection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_e2:
    if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
        with open(history_file, 'w') as f:
            json.dump([], f)
        st.rerun()

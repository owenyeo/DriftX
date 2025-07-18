import streamlit as st
import pandas as pd
import json
from drift.drift_detector import detect_drift, read_logs, calculate_psi


st.set_page_config(page_title="DriftX Dashboard", layout="wide")

st.title("🧠 DriftX Monitoring Dashboard")

# Load logs
try:
    df_logs = read_logs()
    st.success(f"{len(df_logs)} valid logs loaded.")
except Exception as e:
    st.error(f"Failed to load logs: {e}")
    st.stop()

# Show recent inputs
st.subheader("📥 Recent Inputs")
st.dataframe(df_logs.sort_values("timestamp", ascending=False).head(10), use_container_width=True)

# Show feature distributions
st.subheader("📊 Feature Distributions")
for col in df_logs.columns:
    if col == "timestamp":
        continue
    st.line_chart(df_logs[[col]])

# Run PSI drift detection
st.subheader("🚨 Drift Detection (PSI)")
mid = len(df_logs) // 2
baseline = df_logs.iloc[:mid]
current = df_logs.iloc[mid:]

for feature in df_logs.columns:
    if feature == "timestamp":
        continue
    psi = calculate_psi(baseline[feature], current[feature])
    if psi < 0.1:
        status = "✅ No drift"
    elif psi < 0.2:
        status = "⚠️ Slight drift"
    else:
        status = "🚨 Significant drift"
    st.write(f"**{feature}** PSI: `{psi:.3f}` — {status}")

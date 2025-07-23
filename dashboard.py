import streamlit as st
import pandas as pd
import json
from drift.drift_detector import  read_logs, calculate_psi


st.set_page_config(page_title="DriftX Dashboard", layout="wide")

st.title("🧠 DriftX Monitoring Dashboard")

# Load logs
try:
    df_inputs, df_outputs = read_logs()
    st.success(f"{len(df_inputs)} valid logs loaded.")
except Exception as e:
    st.error(f"Failed to load logs: {e}")
    st.stop()

# Show recent inputs
st.subheader("📥 Recent Inputs")
st.dataframe(df_inputs.sort_values("timestamp", ascending=False).head(10), use_container_width=True)

# Show feature distributions
st.subheader("📊 Feature Distributions")
for col in df_inputs.columns:
    if col == "timestamp":
        continue
    st.markdown(f"**{col}**")
    st.line_chart(df_inputs[[col]])

st.subheader("📊 Prediction Distributions")
for col in df_outputs.columns:
    if col == "timestamp":
        continue
    st.markdown(f"**{col}**")
    st.line_chart(df_outputs[[col]])

# Run PSI drift detection
st.subheader("🚨 Drift Detection (PSI)")
mid = len(df_inputs) // 2
baseline = df_inputs.iloc[:mid]
current = df_inputs.iloc[mid:]
st.markdown("**Data side drift**")
for feature in df_inputs.columns:
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

mid = len(df_outputs) // 2
baseline = df_outputs.iloc[:mid]
current = df_outputs.iloc[mid:]
for feature in df_outputs.columns:
    if feature == "timestamp":
        continue
    psi = calculate_psi(baseline[feature], current[feature])
    if psi < 0.1:
        status = "✅ No drift"
    elif psi < 0.2:
        status = "⚠️ Slight drift"
    else:
        status = "🚨 Significant drift"
    st.markdown("**Model side drift**")
    st.write(f"**{feature}** PSI: `{psi:.3f}` — {status}")

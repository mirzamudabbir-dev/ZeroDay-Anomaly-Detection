import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
TEST_LOGS_PATH = "/Users/mudabbir/Documents/SIH/dataset/test/test_logs.csv"
HISTOGRAM_PATH = "/Users/mudabbir/Documents/SIH/dataset/test/anomaly_score_histogram.png"
SCATTER_PATH = "/Users/mudabbir/Documents/SIH/dataset/test/anomaly_score_scatter_topN.png"

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.title("🔍 Anomaly Detection Results Dashboard")

# Load test logs
if os.path.exists(TEST_LOGS_PATH):
    df = pd.read_csv(TEST_LOGS_PATH)
    st.success(f"Loaded {len(df)} records from test logs ✅")

    # Show summary
    st.subheader("📊 Summary")
    st.write(df['label'].value_counts())

    # Histogram from file
    st.subheader("📈 Anomaly Score Distribution")
    if os.path.exists(HISTOGRAM_PATH):
        st.image(HISTOGRAM_PATH, caption="Anomaly Score Histogram")
    else:
        st.warning("Histogram not found. Run evaluate_model.py first.")

    # Scatter plot from file
    st.subheader("📉 Top Anomalies Scatter")
    if os.path.exists(SCATTER_PATH):
        st.image(SCATTER_PATH, caption="Scatter Plot of Top Anomalies")
    else:
        st.warning("Scatter plot not found. Run evaluate_model.py first.")

    # Interactive Table
    st.subheader("🔎 Explore Data")
    option = st.selectbox("Filter by label:", ["All", "Anomalous", "Normal"])
    if option != "All":
        filtered = df[df['label'] == option]
    else:
        filtered = df
    st.dataframe(filtered.head(100))  # Show first 100 rows

    # Top anomalies
    st.subheader("🔥 Top 20 Anomalies")
    top_anomalies = df.sort_values(by="anomaly_score", ascending=False).head(20)
    st.dataframe(top_anomalies)

    # Download option
    st.download_button(
        label="📥 Download Test Logs CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="test_logs.csv",
        mime="text/csv"
    )

else:
    st.error("❌ Test logs not found. Run evaluate_model.py first.")
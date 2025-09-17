import streamlit as st
import pandas as pd
import torch
import os
import requests

# -----------------------------
# 1️⃣ Model Download & Load
# -----------------------------
MODEL_URL = "https://huggingface.co/mirzamudabbir-dev/zeroday-anomaly-model/resolve/main/contrastive_model_ntxent.pth"  # Replace with your Hugging Face raw model URL
MODEL_PATH = "contrastive_model_ntxent.pth"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    st.success("Model downloaded.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# -----------------------------
# 2️⃣ App UI
# -----------------------------
st.title("Zero-Day Anomaly Detection Dashboard")

st.markdown("""
Upload CSV files containing network or system logs to detect anomalies.
You can also enter a site URL for live analysis (feature placeholder for now).
""")

# CSV Upload
uploaded_file = st.file_uploader("Upload CSV for anomaly detection", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # -----------------------------
    # 3️⃣ Preprocess & Model Inference
    # -----------------------------
    try:
        # Replace this with your actual preprocessing
        processed_data = torch.tensor(df.select_dtypes('float').values.astype('float32'))

        # Model inference
        with torch.no_grad():
            predictions = model(processed_data)

        # Display predictions
        st.subheader("Predictions / Anomaly Scores")
        st.dataframe(predictions.numpy())
    except Exception as e:
        st.error(f"Error during inference: {e}")

# Site URL input (placeholder)
site_url = st.text_input("Enter site URL to analyze")
if site_url:
    st.write(f"Analysis results for {site_url} will appear here (feature under development).")

# -----------------------------
# 4️⃣ Analytics & Plots
# -----------------------------
if uploaded_file:
    st.subheader("Anomaly Score Histogram")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(predictions.numpy(), bins=30)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Plotting error: {e}")

st.info("Dashboard is ready. Upload CSV or enter URL to start analysis.")
import streamlit as st
from pathlib import Path

# Set Streamlit page configuration
st.set_page_config(
    page_title="LogSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main App Content ---
st.title("🛡️ LogSentinel: Edge-Optimized Anomaly Detection")
st.markdown("---")

st.header("Welcome to LogSentinel")

st.write("""
This application provides a complete toolkit for training, evaluating, and deploying
log anomaly detection models based on the LogLLM architecture, optimized for
resource-constrained environments.

**Navigate using the sidebar to:**
- **Train & Evaluate:** Launch a new training run with custom models and datasets.
- **View History:** Review detailed reports and metrics from past training runs.
- **Inference:** Perform real-time anomaly detection on log sequences.
""")

# --- Instructions for setup ---
st.info("""
**Getting Started:**
1.  Ensure you have placed your datasets in the `datasets/` directory. Each dataset should have its own folder (e.g., `datasets/BGL/`) containing `train.csv` and `test.csv`.
2.  Pre-downloaded models can be placed in the `models/` directory.
3.  Use the `run_training.py` script for command-line training or use the **Train & Evaluate** page to start a run from the UI.
""")

# --- Hide Streamlit default footer ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
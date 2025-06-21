import streamlit as st
from pathlib import Path
import shutil
import os
from config import DB_PATH, REPORTS_DIR, DATA_CACHE_DIR # Import new central data dir

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
1.  Ensure you have placed your datasets in the `datasets/` directory. Each dataset should have its own folder (e.g., `datasets/BGL/`) containing `train.csv`, `validation.csv`, and `test.csv`.
2.  Pre-downloaded models can be placed in the `models/` directory.
3.  Use the `run_training.py` script for command-line training or use the **Train & Evaluate** page to start a run from the UI.
""")


# --- Danger Zone for Application Reset ---
st.markdown("---")
with st.expander("⚠️ Danger Zone: Application Reset"):
    st.warning(
        "**This is a destructive action and cannot be undone.**\n\n"
        "Clicking this button will permanently delete:\n"
        "- All run history from the database (`logsentinel.db`).\n"
        "- All generated reports and plots from the `reports/` directory.\n"
        "- The entire `logsentinel_data/` directory, including all cached embeddings and temporary models."
    )

    if st.button("Factory Reset Application", type="primary", use_container_width=True):
        try:
            # Delete database file
            if DB_PATH.exists():
                os.remove(DB_PATH)
                st.toast(f"Deleted database: {DB_PATH}", icon="🗑️")

            # Delete reports directory
            if REPORTS_DIR.exists() and REPORTS_DIR.is_dir():
                shutil.rmtree(REPORTS_DIR)
                st.toast(f"Deleted reports directory: {REPORTS_DIR}", icon="🗑️")

            # --- FIX: Delete the entire centralized data directory ---
            if DATA_CACHE_DIR.exists() and DATA_CACHE_DIR.is_dir():
                shutil.rmtree(DATA_CACHE_DIR)
                st.toast(f"Deleted data directory: {DATA_CACHE_DIR}", icon="🗑️")
            
            # Re-create directories for future runs
            REPORTS_DIR.mkdir(exist_ok=True)
            DATA_CACHE_DIR.mkdir(exist_ok=True)
            (DATA_CACHE_DIR / 'embedding_cache').mkdir(exist_ok=True)
            (DATA_CACHE_DIR / 'temp_models').mkdir(exist_ok=True)
            
            st.success("✅ Application has been successfully reset!")
            st.info("Please refresh the page to see the changes.")
            
        except Exception as e:
            st.error(f"An error occurred during reset: {e}")
            print(f"CRITICAL ERROR during Factory Reset: {e}")
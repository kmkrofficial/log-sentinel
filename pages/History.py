import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.database_manager import DatabaseManager
from config import DB_PATH, EXECUTIONS_DIR

st.set_page_config(
    page_title="Run History",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Run History")

db = DatabaseManager(DB_PATH)

try:
    runs_df = db.get_all_runs()
except pd.errors.DatabaseError as e:
    st.error(f"Database error: {e}. Deleting and recreating database...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    db = DatabaseManager(DB_PATH)
    runs_df = db.get_all_runs()

if runs_df.empty:
    st.info("No runs found in the database. Start a new run on the 'Train & Evaluate' page.")
    st.stop()

# Note: st.dataframe still uses use_container_width as of current versions.
# The general warning may not apply to this specific widget yet.
st.dataframe(runs_df, use_container_width=True)

st.header("View Run Details")
run_options = {f"ID: {row['id']} - {row['nickname']} ({row['status']})": row['id'] for _, row in runs_df.iterrows()}
selected_option = st.selectbox("Select a run to view details", options=run_options.keys())

if selected_option:
    selected_run_id = run_options[selected_option]
    details = db.get_run_details(selected_run_id)

    if details:
        st.subheader(f"Details for Run ID: {details['id']} ({details['nickname']})")

        model_metrics = {
            "Accuracy": details.get('accuracy'),
            "Precision": details.get('precision'),
            "F1-Score": details.get('f1_score'),
            "Recall": details.get('recall')
        }

        hardware_metrics = {
            "Total Run Time": f"{(details.get('total_run_time_sec') or 0):.2f}s",
            "Training Time": f"{(details.get('training_time_sec') or 0):.2f}s",
            "Testing Time": f"{(details.get('testing_time_sec') or 0):.2f}s",
            "Avg RAM Usage": f"{(details.get('avg_ram_usage_gb') or 0):.2f} GB",
            "95th Peak RAM": f"{(details.get('peak_95_ram_usage_gb') or 0):.2f} GB",
            "Avg VRAM Usage": f"{(details.get('avg_gpu_vram_gb') or 0):.2f} GB",
            "95th Peak VRAM": f"{(details.get('peak_95_gpu_vram_gb') or 0):.2f} GB",
        }

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", details['status'])
            st.metric("Dataset", details['dataset_name'])
            st.metric("F1-Score", f"{(details.get('f1_score') or 0):.4f}")
            st.subheader("Model Metrics")
            st.json(model_metrics)

        with col2:
            st.metric("Start Time", details['start_time'])
            st.metric("Model", details['model_name'].split('/')[-1])
            st.metric("Total Time", f"{(details.get('total_run_time_sec') or 0):.2f}s")
            st.subheader("Hardware Metrics")
            st.json(hardware_metrics)

        with st.expander("Hyperparameters"):
            st.json(details['hyperparameters'])

        if details['report_path'] and Path(details['report_path']).exists():
            st.subheader("Visualizations")
            viz_dir = Path(details['report_path']) / "visualizations"
            if viz_dir.exists():
                images = [f for f in viz_dir.glob("*.png")]
                for img_path in images:
                    st.image(str(img_path), caption=img_path.name)
            else:
                st.warning("Visualizations directory not found.")
        else:
            st.warning("Report path not found or is inaccessible.")
    else:
        st.error("Could not retrieve run details.")
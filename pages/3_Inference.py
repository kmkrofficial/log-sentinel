import streamlit as st
import pandas as pd
import threading
import time
import os
import shutil
import tempfile
from pathlib import Path

from utils.database_manager import DatabaseManager
from utils.global_state import GLOBAL_APP_STATE, APP_LOCK
from engine.inference_controller import InferenceController
from config import REPORTS_DIR, DB_PATH
from utils.ui_helpers import reset_global_state, callback_handler, render_run_status

st.set_page_config(page_title="Evaluate & Predict", page_icon="🧪", layout="wide")

def get_trained_models():
    """Finds all valid, trained models in the reports directory."""
    if not REPORTS_DIR.is_dir():
        return []
    return [
        d.name for d in REPORTS_DIR.iterdir()
        if d.is_dir() and (d / 'final_model' / 'Llama_ft').exists()
    ]

def run_evaluation_in_thread(trained_run_id, temp_file_path, batch_size, mode):
    db_manager = None
    try:
        db_manager = DatabaseManager(DB_PATH)
        controller = InferenceController(
            trained_run_id=trained_run_id,
            db_manager=db_manager,
            callback=callback_handler
        )
        # The controller now handles DB record creation and all logic
        controller.run(
            input_file_path=temp_file_path,
            mode=mode,
            internal_batch_size=batch_size
        )
    except Exception as e:
        print(f"Error during {mode} thread: {e}")
        GLOBAL_APP_STATE["error"] = str(e)
    finally:
        if db_manager:
            db_manager.close()
        with APP_LOCK:
            GLOBAL_APP_STATE["is_task_running"] = False
        GLOBAL_APP_STATE["done"] = True

st.title("🧪 Evaluate & Predict with a Trained Model")
st.markdown("---")

is_any_task_running = GLOBAL_APP_STATE.get("is_task_running")
if is_any_task_running:
    st.warning(f"A '{GLOBAL_APP_STATE.get('task_type')}' task is currently running. All controls are disabled.")

col1, col2 = st.columns(2)

with col1:
    st.header("Configuration")
    
    st.subheader("1. Select Model")
    trained_models = get_trained_models()
    if not trained_models:
        st.error("No trained models found. Please train a model first on the 'Train & Evaluate' page.")
        st.stop()
    selected_run_id = st.selectbox("Select a Trained Model", options=trained_models, disabled=is_any_task_running, label_visibility="collapsed")
    
    st.subheader("2. Upload Data")
    st.info("Upload a CSV file containing a 'Content' column with log sequences.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", disabled=is_any_task_running)
    
    st.subheader("3. Select Mode")
    evaluate_mode = st.checkbox("Evaluate with detailed metrics (requires 'Label' column in CSV)", value=False, disabled=is_any_task_running)
    mode_str = "testing" if evaluate_mode else "inference"
    task_type_str = "Testing" if evaluate_mode else "Inference"

    batch_size = st.number_input("Inference Batch Size", min_value=1, max_value=512, value=32, help="Number of sequences to process at once. Larger is faster but uses more VRAM.", disabled=is_any_task_running)
    
    st.subheader("4. Launch Run")
    if st.button(f"🚀 Launch {task_type_str} Run", type="primary", use_container_width=True, disabled=is_any_task_running):
        if not selected_run_id or not uploaded_file:
            st.warning("Please select a model and upload a file.")
        else:
            reset_global_state()
            with APP_LOCK:
                GLOBAL_APP_STATE["is_task_running"] = True
                GLOBAL_APP_STATE["task_type"] = task_type_str
            
            # Persist the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state.temp_file_path = tmp.name
            
            thread = threading.Thread(
                target=run_evaluation_in_thread,
                args=(selected_run_id, st.session_state.temp_file_path, batch_size, mode_str)
            )
            thread.start()
            st.rerun()

    # --- Handle results from inference mode ---
    result_from_thread = GLOBAL_APP_STATE.get("latest_progress", {}).get("result")
    if result_from_thread and isinstance(result_from_thread, str):
        st.success("Inference completed!")
        with open(result_from_thread, "rb") as fp:
            st.download_button(
                "Download Results CSV",
                fp,
                f"inference_results_{Path(result_from_thread).stem.split('_')[-1]}.csv",
                "text/csv"
            )
        reset_global_state() # Clear the state after handling the result

with col2:
    st.header("Live Run Status")
    task_type = GLOBAL_APP_STATE.get("task_type")
    # Conditionally render status for either task type, as this page now handles both
    if task_type in ["Inference", "Testing"]:
        render_run_status(task_type)
    else:
        st.info("Status of a run will be displayed here once started.")
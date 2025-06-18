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

def get_trained_models_from_db():
    """Fetches all completed training runs from the database."""
    db = DatabaseManager(DB_PATH)
    # Fetch only completed training runs
    training_runs = db.get_all_runs(run_type='Training')
    db.close()
    
    # Filter for runs that have a final model saved
    valid_models = [
        run for run in training_runs 
        if run['status'] == 'COMPLETED' and (REPORTS_DIR / str(run['run_id']) / 'final_model').exists()
    ]
    return valid_models

def run_evaluation_in_thread(trained_run_id, temp_file_path, batch_size, mode):
    db_manager = None
    try:
        db_manager = DatabaseManager(DB_PATH)
        controller = InferenceController(
            trained_run_id=trained_run_id,
            db_manager=db_manager,
            callback=callback_handler
        )
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

trained_models = get_trained_models_from_db()
col1, col2 = st.columns(2)

with col1:
    st.header("Configuration")
    
    st.subheader("1. Select Model")
    if not trained_models:
        st.error("No trained models found. Please train a model first on the 'Train & Evaluate' page.")
        st.stop()

    # Create a list of nicknames for the selectbox
    model_options = {run['nickname']: run['run_id'] for run in trained_models}
    selected_nickname = st.selectbox("Select a Trained Model by Nickname", options=list(model_options.keys()), disabled=is_any_task_running, label_visibility="collapsed")
    selected_run_id = model_options.get(selected_nickname)

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
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state.temp_file_path = tmp.name
            
            thread = threading.Thread(
                target=run_evaluation_in_thread,
                args=(selected_run_id, st.session_state.temp_file_path, batch_size, mode_str)
            )
            thread.start()
            st.rerun()

    result_from_thread = GLOBAL_APP_STATE.get("latest_progress", {}).get("result")
    if result_from_thread and isinstance(result_from_thread, str):
        st.success("Inference completed!")
        with open(result_from_thread, "rb") as fp:
            st.download_button("Download Results CSV", fp, f"inference_results_{Path(result_from_thread).stem.split('_')[-1]}.csv", "text/csv")
        reset_global_state()

with col2:
    st.header("Live Run Status")
    task_type = GLOBAL_APP_STATE.get("task_type")
    if task_type in ["Inference", "Testing"]:
        render_run_status(task_type)
    else:
        st.info("Status of a run will be displayed here once started.")
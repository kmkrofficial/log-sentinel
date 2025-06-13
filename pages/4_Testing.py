import streamlit as st
import pandas as pd
import threading
import time
import os
from pathlib import Path

from utils.database_manager import DatabaseManager
from utils.global_state import GLOBAL_APP_STATE, APP_LOCK
from engine.testing_controller import TestingController
from config import DB_PATH, DATA_DIR

st.set_page_config(page_title="Testing", page_icon="🧪", layout="wide")

def format_time(seconds):
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0: return "Calculating..."
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def reset_global_state():
    GLOBAL_APP_STATE["is_task_running"] = False
    GLOBAL_APP_STATE["task_type"] = None
    GLOBAL_APP_STATE["log_buffer"] = []
    GLOBAL_APP_STATE["latest_progress"] = {}
    GLOBAL_APP_STATE["stop_requested"] = False
    GLOBAL_APP_STATE["result_buffer"] = None
    GLOBAL_APP_STATE["error"] = None
    GLOBAL_APP_STATE["done"] = False

def callback_handler(data):
    if GLOBAL_APP_STATE["stop_requested"]: return "STOP"
    GLOBAL_APP_STATE["latest_progress"] = data
    if "log" in data: GLOBAL_APP_STATE["log_buffer"].append(data["log"])
    if "error" in data: GLOBAL_APP_STATE["error"] = data["error"]
    return "CONTINUE"

def run_testing_in_thread(trained_run_id, test_file_path):
    controller = None
    with APP_LOCK:
        if GLOBAL_APP_STATE.get("is_task_running"):
            GLOBAL_APP_STATE["error"] = "Another task is already running."
            return
        reset_global_state()
        GLOBAL_APP_STATE["is_task_running"] = True
        GLOBAL_APP_STATE["task_type"] = "Testing"
    db_manager = DatabaseManager(DB_PATH)
    controller = TestingController(
        trained_run_id=trained_run_id, test_filename=test_file_path,
        db_manager=db_manager, callback=callback_handler
    )
    try:
        controller.run()
    except Exception as e:
        print(f"Error in testing thread: {e}")
        GLOBAL_APP_STATE["error"] = str(e)
    finally:
        if controller: controller.cleanup()
        with APP_LOCK: GLOBAL_APP_STATE["is_task_running"] = False
        db_manager.close()
        GLOBAL_APP_STATE["done"] = True

st.title("🧪 Test a Trained Model")
st.markdown("---")

is_any_task_running = GLOBAL_APP_STATE.get("is_task_running")
if is_any_task_running:
    st.warning(f"A '{GLOBAL_APP_STATE.get('task_type')}' task is currently running. All controls are disabled until it is complete.")

col1, col2 = st.columns(2) # Changed to equal width columns

with col1:
    st.header("Test Configuration")
    db = DatabaseManager(DB_PATH)
    all_runs = db.get_all_runs()
    db.close()
    if not all_runs:
        st.warning("No runs found in DB. Please complete a training run first.")
        st.stop()
    
    runs_df = pd.DataFrame(all_runs)
    completed_runs = runs_df[(runs_df['run_type'] == 'Training') & (runs_df['status'] == 'COMPLETED')].copy()

    if completed_runs.empty:
        st.warning("No successfully completed training runs available to test.")
        st.stop()

    run_options = {f"{row['run_id'][:8]}... ({row['model_name']})": row['run_id'] for _, row in completed_runs.iterrows()}
    selected_run_display = st.selectbox("1. Select a Trained Model to Test", options=run_options.keys(), disabled=is_any_task_running)
    selected_run_id = run_options[selected_run_display] if selected_run_display else None
    
    st.subheader("2. Upload a Test File")
    st.info("The file must be a CSV with 'Content' and 'Label' columns.")
    uploaded_file = st.file_uploader("Upload test.csv", type=["csv"], label_visibility="collapsed", disabled=is_any_task_running)
    
    st.subheader("3. Launch Run")
    if st.button("🚀 Launch Test Run", type="primary", use_container_width=True, disabled=is_any_task_running):
        if not selected_run_id or not uploaded_file:
            st.warning("Please select a trained model and upload a test file.")
        else:
            temp_dir = DATA_DIR / "temp"; temp_dir.mkdir(exist_ok=True)
            test_file_path = temp_dir / uploaded_file.name
            with open(test_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            thread = threading.Thread(target=run_testing_in_thread, args=(selected_run_id, str(test_file_path)))
            thread.start()
            st.rerun()

with col2:
    st.header("Live Run Status")
    is_this_task_running = is_any_task_running and GLOBAL_APP_STATE.get("task_type") == "Testing"
    if is_this_task_running:
        progress_info = GLOBAL_APP_STATE.get("latest_progress", {})
        progress = progress_info.get("progress", 0)
        rows_processed = progress_info.get("rows_processed", 0)
        etc = progress_info.get("etc", 0)
        
        st.text(f"{progress:.1%}")
        st.progress(progress)
        st.text(f"Rows Processed: {rows_processed:,}")
        st.text(f"ETC: {format_time(etc)}")
        
        if st.button("Stop Test Run", use_container_width=True):
            GLOBAL_APP_STATE["stop_requested"] = True
            st.warning("Stop request sent.")
        
        with st.expander("Show Live Logs", expanded=True):
            st.code('\n'.join(GLOBAL_APP_STATE.get("log_buffer", [])), language='log', height=500)
        
        if GLOBAL_APP_STATE.get("error"): st.error(f"An error occurred: {GLOBAL_APP_STATE['error']}")
        time.sleep(1)
        st.rerun()
    else:
        st.info("Status of the run will be displayed here.")
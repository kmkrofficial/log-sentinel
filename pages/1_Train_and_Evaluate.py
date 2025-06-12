import streamlit as st
import pandas as pd
import sys
import os
import json
import time
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine.training_controller import TrainingController
from utils.database_manager import DatabaseManager
from utils.model_loader import get_local_models
from config import DEFAULT_HYPERPARAMETERS, DB_PATH, DATA_DIR
# Import the new global state and lock
from utils.global_state import APP_LOCK, TRAINING_STATUS

# --- Helper Functions (Do not use Streamlit objects) ---
def get_available_datasets():
    if not DATA_DIR.is_dir(): return []
    return [p.name for p in DATA_DIR.iterdir() if p.is_dir() and (p / 'train.csv').exists()]

def training_callback(status):
    with APP_LOCK:
        TRAINING_STATUS["latest_progress"] = status
        if TRAINING_STATUS["stop_requested"]: return 'STOP'
    return 'CONTINUE'

class ThreadLogRedirector:
    def write(self, message):
        if message.strip():
            with APP_LOCK:
                TRAINING_STATUS["log_buffer"].append(message.strip())
    def flush(self): pass

def run_training_in_thread(run_inputs):
    original_stdout = sys.stdout
    sys.stdout = ThreadLogRedirector()
    db_manager = DatabaseManager(DB_PATH)
    controller = TrainingController(
        model_name=run_inputs['model_id'],
        dataset_name=run_inputs['dataset'],
        hyperparameters=json.loads(run_inputs['hp_json']),
        db_manager=db_manager,
        callback=training_callback
    )
    try:
        controller.run()
    except Exception as e:
        print(f"CRITICAL ERROR IN THREAD: {e}")
    finally:
        sys.stdout = original_stdout
        with APP_LOCK:
            TRAINING_STATUS["is_running"] = False
        db_manager.close()

# --- UI Layout ---
st.set_page_config(page_title="Train & Evaluate", layout="wide")
st.title("🚀 Train & Evaluate a New Model")
st.markdown("---")

with APP_LOCK:
    is_training_globally = TRAINING_STATUS["is_running"]

if is_training_globally:
    st.info("A training run is already in progress application-wide. Please wait for it to complete.")

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Configuration")
    with st.form("training_form"):
        st.subheader("1. Select Model")
        local_models, hf_model_id = get_local_models(), st.text_input("Hugging Face Model ID", "princeton-nlp/Sheared-Llama-1.3B")
        model_to_use = st.selectbox("Local model", [""] + local_models, index=0) or hf_model_id
        
        st.subheader("2. Select Dataset")
        dataset_to_use = st.selectbox("Dataset", get_available_datasets(), index=0)
        
        st.subheader("3. Hyperparameters")
        hp_json_str = st.text_area("JSON", json.dumps(DEFAULT_HYPERPARAMETERS, indent=4), height=300)

        submitted = st.form_submit_button("Start Training Run", type="primary", disabled=is_training_globally)
        if submitted:
            try:
                json.loads(hp_json_str)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()
            
            with APP_LOCK:
                TRAINING_STATUS.update({"is_running": True, "stop_requested": False, "log_buffer": ["--- Starting new run... ---"], "latest_progress": {}})
            
            run_inputs = {"model_id": model_to_use, "dataset": dataset_to_use, "hp_json": hp_json_str}
            thread = threading.Thread(target=run_training_in_thread, args=(run_inputs,), daemon=True)
            thread.start()
            st.rerun()

with col2:
    st.header("Run Progress")
    if is_training_globally:
        if st.button("Stop Run", type="secondary"):
            with APP_LOCK:
                TRAINING_STATUS["stop_requested"] = True
            st.warning("Stop request sent. The run will abort after the current step.")

        with APP_LOCK:
            progress_info = TRAINING_STATUS.get("latest_progress", {})
            log_messages = TRAINING_STATUS.get("log_buffer", [])
        
        epoch, progress = progress_info.get("epoch", "Initializing..."), progress_info.get("progress", 0.0)
        loss, etc = progress_info.get("loss", 0.0), progress_info.get("etc", 0)
        
        def format_time(s):
            if s is None or s < 0: return "N/A"
            h, rem = divmod(int(s), 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        c1, c2 = st.columns(2)
        c1.text(f"Current Phase: {epoch}")
        c2.text(f"Est. Time Remaining: {format_time(etc)}")
        st.progress(progress, text=f"Overall Progress: {progress:.1%} | Last Batch Loss: {loss:.4f}")
        
        st.subheader("Live Console Log")
        log_container = st.container(height=500, border=True)
        log_container.text("\n".join(log_messages))
        
        time.sleep(2)
        st.rerun()
    else:
        st.info("Start a run to see live progress and logs.")
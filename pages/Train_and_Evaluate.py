import streamlit as st
from pathlib import Path
import sys
import os
import threading
import queue
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.training_controller import TrainingController
from utils.database_manager import DatabaseManager
from config import (
    DATA_DIR, DB_PATH, MODELS_DIR, 
    DEFAULT_LLAMA_MODEL, DEFAULT_ENCODER_MODEL
)
from utils.ui_helpers import (
    get_dataset_options, 
    render_progress_bar, render_metrics, render_logs, 
    VRAM_WARNING_THRESHOLD, RAM_WARNING_THRESHOLD
)
from utils.global_state import GlobalState
from system_spec import get_spec_dict

st.set_page_config(
    page_title="Train & Evaluate",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Train & Evaluate Model")

db = DatabaseManager(DB_PATH)
state = GlobalState.get_instance()

def check_models_present():
    llama_path = MODELS_DIR / DEFAULT_LLAMA_MODEL.split('/')[-1]
    encoder_path = MODELS_DIR / DEFAULT_ENCODER_MODEL.split('/')[-1]
    
    llama_exists = llama_path.exists() and any(llama_path.glob("*.safetensors"))
    encoder_exists = encoder_path.exists() and (encoder_path / 'modules.json').exists()
    
    return llama_exists, encoder_exists, llama_path, encoder_path

def start_training_thread(dataset_name, is_test_run, test_run_pct):
    q = queue.Queue()
    state.set_train_state(True, q)
    
    def run():
        try:
            controller = TrainingController(
                dataset_name=dataset_name,
                db_manager=db,
                callback=q.put,
                is_test_run=is_test_run,
                test_run_percentage=test_run_pct
            )
            controller.run()
        except Exception as e:
            q.put({"error": str(e)})
        finally:
            state.set_train_state(False, None)
            q.put({"done": True})

    threading.Thread(target=run, daemon=True).start()

def get_current_specs():
    try:
        return get_spec_dict()
    except Exception:
        return {}

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Configuration")
    
    with st.container(border=True):
        st.subheader("Model Verification")
        llama_found, encoder_found, llama_path, encoder_path = check_models_present()
        
        if llama_found:
            st.success(f"LLM found: `{llama_path.name}`")
        else:
            st.error(f"LLM not found. Expected model files in: `{llama_path}`")
            st.warning("Llama is a gated model. Please run `huggingface-cli login` in your terminal, then run `python download_models.py` to fix this.")
            
        if encoder_found:
            st.success(f"Encoder found: `{encoder_path.name}`")
        else:
            st.error(f"Encoder not found. Expected file: `{encoder_path / 'modules.json'}`")
            st.warning("Please run `python download_models.py` to download the encoder model.")
            
    models_ready = llama_found and encoder_found
    
    dataset_options = get_dataset_options(DATA_DIR)
    if not dataset_options:
        st.error(f"No datasets found in `{DATA_DIR}`. Please add datasets to continue.")
        st.stop()
        
    dataset_name = st.selectbox(
        "Select Dataset",
        options=dataset_options,
        index=0,
        help="Select the dataset to use for training and evaluation. Hyperparameters are set automatically."
    )

    st.subheader("Quick Test Run")
    is_test_run = st.checkbox("Run a quick test", value=False, help="Use a small fraction of the data for a fast test run.")
    test_run_percentage = st.slider("Test Run Data Percentage", min_value=0.01, max_value=1.0, value=0.1, step=0.01, disabled=not is_test_run)


    if st.button("🚀 Start Training", type="primary", disabled=state.is_training or not models_ready, use_container_width=True):
        if dataset_name:
            specs = get_current_specs()
            gpu_vram = specs.get('gpu', {}).get('total_vram_gb', 0)
            total_ram = specs.get('ram', {}).get('total_gb', 0)

            if gpu_vram > 0 and gpu_vram < VRAM_WARNING_THRESHOLD:
                st.warning(f"Low VRAM ({gpu_vram:.1f}GB) detected. Training may be slow or fail.")
            if total_ram < RAM_WARNING_THRESHOLD:
                st.warning(f"Low System RAM ({total_ram:.1f}GB) detected. Data processing may be slow.")

            st.session_state.log_messages = []
            st.session_state.metrics = {}
            st.session_state.progress = 0.0
            st.session_state.status = "Starting..."
            
            start_training_thread(
                dataset_name,
                is_test_run,
                test_run_percentage
            )
        else:
            st.error("Please select a dataset.")

with col2:
    st.header("Training Status")
    
    if state.is_training and state.queue:
        while state.queue and not state.queue.empty():
            msg = state.queue.get()
            if "log" in msg:
                st.session_state.log_messages.insert(0, msg['log'])
            if "status" in msg:
                st.session_state.status = msg['status']
            if "progress" in msg:
                st.session_state.progress = msg['progress']
            if "metrics" in msg:
                st.session_state.metrics.update(msg['metrics'])
            if "validation_metrics" in msg:
                st.session_state.metrics.update(msg['validation_metrics'])
            if "error" in msg:
                st.session_state.status = "Error!"
                st.error(msg['error'])
                state.set_train_state(False, None)
            if "done" in msg and msg['done']:
                if st.session_state.status != "Error!":
                    st.session_state.status = "Training complete."
                st.balloons()

    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    log_placeholder = st.empty()

    render_progress_bar(progress_placeholder)
    render_metrics(metrics_placeholder)
    render_logs(log_placeholder)
    
    if state.is_training:
        time.sleep(1)
        st.rerun()
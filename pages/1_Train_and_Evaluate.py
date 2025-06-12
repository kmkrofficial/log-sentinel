import streamlit as st
import pandas as pd
import sys
import os
from contextlib import contextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine.training_controller import TrainingController
from utils.database_manager import DatabaseManager
from utils.model_loader import get_local_models
from config import DEFAULT_HYPERPARAMETERS, DB_PATH, DATA_DIR

class StreamlitLogCapture:
    def __init__(self, container):
        self.container = container
        self.buffer = ""
    def write(self, message):
        self.buffer += message
        self.container.text(self.buffer)
    def flush(self): pass

@contextmanager
def st_capture_stdout(container):
    original_stdout = sys.stdout
    sys.stdout = StreamlitLogCapture(container)
    try:
        yield
    finally:
        sys.stdout = original_stdout

def get_available_datasets():
    if not DATA_DIR.is_dir(): return []
    return [p.name for p in DATA_DIR.iterdir() if p.is_dir() and (p / 'train.csv').exists()]

st.set_page_config(page_title="Train & Evaluate", layout="wide")
st.title("🚀 Train & Evaluate a New Model")
st.markdown("---")

if 'stop_training' not in st.session_state:
    st.session_state['stop_training'] = False

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuration")
    st.subheader("1. Select Model")
    local_models = get_local_models()
    hf_model_id = st.text_input("Enter Hugging Face Model ID", "princeton-nlp/Sheared-Llama-1.3B")
    model_to_use = st.selectbox("Or choose a local model", [""] + local_models, index=0) or hf_model_id
    st.info(f"**Model to be used:** `{model_to_use}`")

    st.subheader("2. Select Dataset")
    available_datasets = get_available_datasets()
    if not available_datasets:
        st.error("No datasets found in the `datasets` directory.")
        dataset_to_use = None
    else:
        dataset_to_use = st.selectbox("Choose a dataset", available_datasets)
    
    # FIX: Replace static JSON with interactive widgets inside an expander.
    st.subheader("3. Hyperparameters")
    with st.expander("Edit Hyperparameters", expanded=False):
        edited_hp = {}
        
        st.markdown("##### Epochs per Phase")
        c1, c2 = st.columns(2)
        edited_hp['n_epochs_phase1'] = c1.number_input("Phase 1 (Proj)", min_value=0, value=DEFAULT_HYPERPARAMETERS['n_epochs_phase1'])
        edited_hp['n_epochs_phase2'] = c2.number_input("Phase 2 (Cls)", min_value=0, value=DEFAULT_HYPERPARAMETERS['n_epochs_phase2'])
        edited_hp['n_epochs_phase3'] = c1.number_input("Phase 3 (Proj+Cls)", min_value=0, value=DEFAULT_HYPERPARAMETERS['n_epochs_phase3'])
        edited_hp['n_epochs_phase4'] = c2.number_input("Phase 4 (All)", min_value=0, value=DEFAULT_HYPERPARAMETERS['n_epochs_phase4'])

        st.markdown("##### Learning Rates per Phase")
        c1, c2 = st.columns(2)
        edited_hp['lr_phase1'] = c1.number_input("LR Phase 1", min_value=0.0, value=DEFAULT_HYPERPARAMETERS['lr_phase1'], format="%.0e")
        edited_hp['lr_phase2'] = c2.number_input("LR Phase 2", min_value=0.0, value=DEFAULT_HYPERPARAMETERS['lr_phase2'], format="%.0e")
        edited_hp['lr_phase3'] = c1.number_input("LR Phase 3", min_value=0.0, value=DEFAULT_HYPERPARAMETERS['lr_phase3'], format="%.0e")
        edited_hp['lr_phase4'] = c2.number_input("LR Phase 4", min_value=0.0, value=DEFAULT_HYPERPARAMETERS['lr_phase4'], format="%.0e")
        
        st.markdown("##### Batch and Sequence Sizes")
        c1, c2 = st.columns(2)
        edited_hp['batch_size'] = c1.number_input("Effective Batch Size", min_value=1, value=DEFAULT_HYPERPARAMETERS['batch_size'])
        edited_hp['micro_batch_size'] = c2.number_input("Micro Batch Size (per GPU pass)", min_value=1, value=DEFAULT_HYPERPARAMETERS['micro_batch_size'])
        edited_hp['max_content_len'] = c1.number_input("Max Content Length", min_value=1, value=DEFAULT_HYPERPARAMETERS['max_content_len'])
        edited_hp['max_seq_len'] = c2.number_input("Max Sequence Length", min_value=1, value=DEFAULT_HYPERPARAMETERS['max_seq_len'])
        
        st.markdown("##### Oversampling")
        edited_hp['min_less_portion'] = st.slider("Minority Class Target Proportion", min_value=0.0, max_value=1.0, value=DEFAULT_HYPERPARAMETERS['min_less_portion'])

    st.markdown("---")
    start_button = st.button("Start Training Run", disabled=(not model_to_use or not dataset_to_use), type="primary")

with col2:
    st.header("Run Progress")
    progress_area = st.empty()
    
    if start_button:
        st.session_state['stop_training'] = False
        
        with progress_area.container():
            stop_button = st.button("Stop Run", type="secondary")
            if stop_button:
                st.session_state['stop_training'] = True
                st.warning("Stop request sent. The run will abort after the current step.")

            status_text = st.empty()
            progress_bar = st.progress(0)
        
        log_container = st.container(height=500, border=True)
        log_area = log_container.empty()
        log_area.text("Logs will appear here when a run is started...")

        def training_callback(status):
            epoch = status.get("epoch", "Starting...")
            progress = status.get("progress", 0.0)
            loss = status.get("loss", 0.0)
            
            with progress_area.container():
                 status_text.text(f"Current Phase: {epoch} | Last Batch Loss: {loss:.4f}")
                 progress_bar.progress(progress, text=f"{progress:.0%}")

            if st.session_state.get('stop_training', False):
                return 'STOP'
            return 'CONTINUE'

        db_manager = DatabaseManager(DB_PATH)
        controller = TrainingController(
            model_name=model_to_use,
            dataset_name=dataset_to_use,
            hyperparameters=edited_hp, # Pass the edited hyperparameters
            db_manager=db_manager,
            callback=training_callback
        )
        
        if not controller.run_id:
            st.error("Failed to create run. Check console for database errors.")
        else:
            st.session_state['run_id'] = controller.run_id
            
            with st_capture_stdout(log_area):
                try:
                    controller.run()
                    with progress_area.container():
                        if st.session_state.get('stop_training', False):
                            st.warning(f"Run {st.session_state['run_id']} was aborted.")
                        else:
                            status_text.text("Completed!")
                            progress_bar.progress(1.0, "Done!")
                            st.success(f"Run {st.session_state['run_id']} completed successfully!")
                            st.toast("✅ Training Complete!")

                except Exception as e:
                    print(f"CRITICAL ERROR in UI: {e}")
                    st.error(f"Run {st.session_state['run_id']} failed: {e}")
                    st.toast("❌ Training Failed!")
        
        db_manager.close()
        st.session_state['stop_training'] = False
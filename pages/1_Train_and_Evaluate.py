import streamlit as st
import pandas as pd
import sys
import os
import json
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
    
    # FIX: Use an editable JSON text area for hyperparameters.
    st.subheader("3. Hyperparameters")
    default_hp_str = json.dumps(DEFAULT_HYPERPARAMETERS, indent=4)
    hp_json_str = st.text_area(
        "Edit Hyperparameters as JSON",
        value=default_hp_str,
        height=300
    )

    st.markdown("---")
    start_button = st.button("Start Training Run", disabled=(not model_to_use or not dataset_to_use), type="primary")

with col2:
    st.header("Run Progress")
    progress_area = st.empty()
    
    if start_button:
        # Validate the JSON before proceeding
        try:
            edited_hp = json.loads(hp_json_str)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in hyperparameters: {e}")
            st.stop() # Halt execution if JSON is invalid

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
            hyperparameters=edited_hp, # Pass the validated, edited hyperparameters
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
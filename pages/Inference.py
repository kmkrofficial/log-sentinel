import streamlit as st
from pathlib import Path
import sys
import os
import threading
import queue
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.inference_controller import InferenceController
from utils.database_manager import DatabaseManager
from config import DATA_DIR, DB_PATH, EXECUTIONS_DIR
from utils.ui_helpers import get_dataset_options, render_logs
from utils.global_state import GlobalState

st.set_page_config(
    page_title="Inference",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Perform Inference")

db = DatabaseManager(DB_PATH)
state = GlobalState.get_instance()

def get_trained_model_options():
    if not EXECUTIONS_DIR.exists():
        return []

    options = []
    for d in EXECUTIONS_DIR.iterdir():
        if d.is_dir() and (d / 'output_model').exists():
            try:
                # Attempt to find run details from nickname for a friendlier name
                details = db.get_runs_by_nickname_prefix(d.name)
                if details:
                     # Use the first match if found
                    options.append((details[0], d))
                else:
                    options.append((d.name, d))
            except Exception:
                options.append((d.name, d))

    options.sort(key=lambda x: x[0], reverse=True)
    return options

def start_inference_thread(model_path, dataset_name, output_filename, is_test_run, test_run_pct):
    q = queue.Queue()
    state.set_train_state(True, q)

    def run():
        try:
            controller = InferenceController(
                model_run_path=model_path,
                dataset_name=dataset_name,
                output_filename=output_filename,
                callback=q.put,
                is_test_run=is_test_run,
                test_run_percentage=test_run_pct
            )
            controller.run_inference()
        except Exception as e:
            q.put({"error": str(e)})
        finally:
            state.set_train_state(False, None)
            q.put({"done": True})

    threading.Thread(target=run, daemon=True).start()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Configuration")

    model_options = get_trained_model_options()
    if not model_options:
        st.error(f"No trained models found in `{EXECUTIONS_DIR}`. Please train a model first.")
        st.stop()

    selected_model_name = st.selectbox(
        "Select Trained Model",
        options=[name for name, path in model_options],
        index=0,
        help="Select a previously trained model from an execution run."
    )

    model_path = dict(model_options)[selected_model_name]

    dataset_options = get_dataset_options(DATA_DIR)
    if not dataset_options:
        st.error(f"No datasets found in `{DATA_DIR}`. Please add datasets to continue.")
        st.stop()

    dataset_name = st.selectbox(
        "Select Dataset for Inference",
        options=dataset_options,
        index=0,
        help="Select the dataset to run inference on. This should be a test set."
    )

    output_filename = st.text_input(
        "Output Filename",
        value=f"predictions_{selected_model_name}_{dataset_name}.csv",
        help="The name of the CSV file to save predictions to. It will be saved in the model's execution directory."
    )

    st.subheader("Quick Test Run")
    is_test_run = st.checkbox("Run a quick test on a fraction of data", value=False)
    test_run_percentage = st.slider("Test Run Data Fraction", min_value=0.01, max_value=1.0, value=0.1, step=0.01, disabled=not is_test_run)

    if st.button("🚀 Start Inference", type="primary", disabled=state.is_training):
        if dataset_name and selected_model_name and output_filename:
            st.session_state.log_messages = []
            st.session_state.status = "Starting..."

            start_inference_thread(
                model_path,
                dataset_name,
                output_filename,
                is_test_run,
                test_run_percentage
            )
        else:
            st.error("Please fill in all fields.")

with col2:
    st.header("Inference Status")

    if state.is_training and state.queue:
        while state.queue and not state.queue.empty():
            msg = state.queue.get()
            if "log" in msg:
                st.session_state.log_messages.insert(0, msg['log'])
            if "status" in msg:
                st.session_state.status = msg['status']
            if "error" in msg:
                st.session_state.status = "Error!"
                st.error(msg['error'])
                state.set_train_state(False, None)
            if "done" in msg and msg['done']:
                if st.session_state.status != "Error!":
                    st.session_state.status = "Inference complete."
                st.balloons()

    status_placeholder = st.empty()
    log_placeholder = st.empty()

    status_placeholder.text(st.session_state.get('status', 'Idle'))
    render_logs(log_placeholder)

    if state.is_training:
        time.sleep(1)
        st.rerun()
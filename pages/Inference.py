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
from utils.ui_helpers import get_dataset_options, render_logs, render_metrics
from utils.global_state import GlobalState

st.set_page_config(
    page_title="Inference",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Perform Inference & Evaluation")

db = DatabaseManager(DB_PATH)
state = GlobalState.get_instance()

def get_trained_model_options():
    if not EXECUTIONS_DIR.exists():
        return {}
    
    options = {}
    try:
        runs_df = db.get_all_runs()
        completed_runs = runs_df[runs_df['status'] == 'COMPLETED'].copy()

        for _, row in completed_runs.iterrows():
            nickname = row.get('nickname')
            if not nickname or nickname.startswith('Inference_'):
                continue

            model_path = EXECUTIONS_DIR / nickname
            if model_path.exists() and (model_path / 'output_model').exists():
                f1_score = row.get('f1_score', 0.0)
                run_id = row.get('id', 'N/A')
                
                label = f"ID {run_id} - {nickname} (F1: {f1_score:.4f})"
                options[label] = str(model_path)
                
    except Exception as e:
        st.error(f"Error loading model options: {e}")
        return {}
        
    sorted_options = dict(sorted(options.items(), key=lambda item: int(item[0].split(" ")[1]), reverse=True))
    return sorted_options


def start_inference_thread(model_run_path, dataset_name, db_manager, is_test_run, test_run_pct, manual_nickname=None):
    q = queue.Queue()
    state.set_train_state(True, q)

    def run():
        try:
            controller = InferenceController(
                model_run_path=model_run_path,
                dataset_name=dataset_name,
                db_manager=db_manager,
                callback=q.put,
                is_test_run=is_test_run,
                test_run_percentage=test_run_pct,
                manual_nickname=manual_nickname
            )
            controller.run_inference()
        except Exception as e:
            q.put({"error": str(e)})
        finally:
            state.set_train_state(False, None)
            q.put({"done": True})

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Configuration")

    model_source_mode = st.radio(
        "Select Model Source",
        ("Select from Run History", "Import External Model"),
        horizontal=True
    )
    
    model_run_path = None
    manual_nickname = None

    if model_source_mode == "Select from Run History":
        model_options = get_trained_model_options()
        if not model_options:
            st.warning("No completed training runs found in the database. Please train a model first, or switch to 'Import External Model'.")
        else:
            selected_model_label = st.selectbox(
                "Select Trained Model",
                options=model_options.keys(),
                index=0
            )
            model_run_path = model_options.get(selected_model_label)
    
    else: # Import External Model
        st.info("The path must point to a directory that contains an `output_model` subdirectory.")
        external_path_str = st.text_input("Absolute Path to Model Directory", "")
        manual_nickname = st.text_input("Provide a Nickname for this Inference Run", "ImportedModel")
        
        if external_path_str:
            path_obj = Path(external_path_str)
            if not path_obj.exists() or not path_obj.is_dir():
                st.error("The provided path does not exist or is not a directory.")
            elif not (path_obj / "output_model").exists():
                st.error("A valid model directory must contain an `output_model` subdirectory.")
            else:
                st.success("Valid model directory found.")
                model_run_path = external_path_str
        
        if not manual_nickname:
            st.warning("Please provide a nickname for this run.")


    dataset_options = get_dataset_options(DATA_DIR)
    if not dataset_options:
        st.error(f"No datasets found in `{DATA_DIR}`. Please add datasets to continue.")
        st.stop()

    dataset_name = st.selectbox(
        "Select Dataset for Evaluation",
        options=dataset_options,
        index=0,
        help="The 'test.csv' from this dataset will be used for evaluation."
    )

    st.subheader("Quick Test Run")
    is_test_run = st.checkbox("Run a quick test on a fraction of data", value=False)
    test_run_percentage = st.slider("Test Run Data Fraction", min_value=0.01, max_value=1.0, value=0.1, step=0.01, disabled=not is_test_run)

    is_ready = model_run_path and dataset_name
    if model_source_mode == "Import External Model" and not manual_nickname:
        is_ready = False

    if st.button("🚀 Start Inference", type="primary", disabled=state.is_training or not is_ready):
        st.session_state.log_messages = []
        st.session_state.metrics = {}
        st.session_state.status = "Starting..."

        start_inference_thread(
            model_run_path,
            dataset_name,
            db,
            is_test_run,
            test_run_percentage,
            manual_nickname
        )

with col2:
    st.header("Inference Status")

    if state.is_training and state.queue:
        while state.queue and not state.queue.empty():
            msg = state.queue.get()
            if "log" in msg:
                st.session_state.log_messages.insert(0, msg['log'])
            if "status" in msg:
                st.session_state.status = msg['status']
            if "validation_metrics" in msg:
                st.session_state.metrics.update(msg['validation_metrics'])
            if "error" in msg:
                st.session_state.status = "Error!"
                st.error(msg['error'])
                state.set_train_state(False, None)
            if "done" in msg and msg['done']:
                if st.session_state.status != "Error!":
                    st.session_state.status = "Inference complete."
                st.balloons()

    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    log_placeholder = st.empty()

    status_placeholder.text(st.session_state.get('status', 'Idle'))
    render_metrics(metrics_placeholder)
    render_logs(log_placeholder)

    if state.is_training:
        time.sleep(1)
        st.rerun()
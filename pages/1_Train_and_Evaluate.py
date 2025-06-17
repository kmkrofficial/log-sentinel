import streamlit as st
import pandas as pd
import threading
import time
import os
import json
from pathlib import Path

from utils.database_manager import DatabaseManager
from utils.global_state import GLOBAL_APP_STATE, APP_LOCK
from engine.training_controller import TrainingController
from config import DEFAULT_HYPERPARAMETERS, DATA_DIR, DB_PATH, MODELS_DIR
from utils.model_loader import get_local_models
from utils.ui_helpers import reset_global_state, callback_handler, render_run_status

st.set_page_config(page_title="Train & Evaluate", page_icon="💪", layout="wide")

def get_available_datasets():
    if not DATA_DIR.is_dir(): return []
    return [d.name for d in DATA_DIR.iterdir() if d.is_dir() and (d / 'train.csv').exists()]

def run_training_in_thread(model_name, dataset_name, hyperparameters):
    db_manager = None
    try:
        db_manager = DatabaseManager(db_path=DB_PATH)
        controller = TrainingController(
            model_name=model_name,
            dataset_name=dataset_name,
            hyperparameters=hyperparameters,
            db_manager=db_manager,
            callback=callback_handler
        )
        controller.run()
    except Exception as e:
        print(f"Error during training thread: {e}")
        GLOBAL_APP_STATE["error"] = str(e)
    finally:
        with APP_LOCK:
            GLOBAL_APP_STATE["is_task_running"] = False
        if db_manager:
            db_manager.close()
        GLOBAL_APP_STATE["done"] = True

st.title("💪 Train & Evaluate a New Model")
st.markdown("---")

is_any_task_running = GLOBAL_APP_STATE.get("is_task_running")
if is_any_task_running:
    st.warning(f"A '{GLOBAL_APP_STATE.get('task_type')}' task is currently running. All controls are disabled.")

rerun_config = st.session_state.pop('rerun_config', None)
col1, col2 = st.columns(2)

with col1:
    st.header("Training Configuration")
    st.subheader("1. Model Selection")
    model_source = st.radio("Select Model Source", ["Hugging Face", "Local"], horizontal=True, index=0, disabled=is_any_task_running)
    
    model_name_input = None
    if model_source == "Hugging Face":
        st.info("""
        **Note:** To use gated models from Hugging Face (like Llama 3), you must first authenticate from your terminal:
        `huggingface-cli login`
        """)
        model_name_input = st.text_input("Enter Hugging Face Model ID", value="meta-llama/Llama-3.2-1B", disabled=is_any_task_running)
    else:
        local_models = get_local_models()
        if not local_models:
            st.warning("No local models found in `models/`.")
            model_name_input = None
        else:
            model_name_input = st.selectbox("Select a Local Model", options=local_models, disabled=is_any_task_running)

    st.subheader("2. Dataset Selection")
    available_datasets = get_available_datasets()
    if not available_datasets:
        st.error("No datasets found in `datasets/`.")
        st.stop()

    dataset_index = 0
    if rerun_config and rerun_config['dataset_name'] in available_datasets:
        dataset_index = available_datasets.index(rerun_config['dataset_name'])
    dataset_name_select = st.selectbox("Select Dataset", options=available_datasets, index=dataset_index, disabled=is_any_task_running)

    st.subheader("3. Hyperparameters")
    initial_hp = rerun_config['hyperparameters'] if rerun_config else DEFAULT_HYPERPARAMETERS
    hp_json_str = json.dumps(initial_hp, indent=4)
    hp_json_input = st.text_area("Edit Hyperparameters (JSON format)", value=hp_json_str, height=300, disabled=is_any_task_running)

    st.subheader("4. Launch Run")
    if st.button("🚀 Launch Training Run", type="primary", use_container_width=True, disabled=is_any_task_running):
        try:
            hyperparams_for_run = json.loads(hp_json_input)
            if not model_name_input:
                st.error("Model name cannot be empty.")
            else:
                reset_global_state()
                with APP_LOCK:
                    GLOBAL_APP_STATE["is_task_running"] = True
                    GLOBAL_APP_STATE["task_type"] = "Training"
                
                thread = threading.Thread(
                    target=run_training_in_thread,
                    args=(model_name_input, dataset_name_select, hyperparams_for_run)
                )
                thread.start()
                st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in hyperparameters: {e}")

with col2:
    st.header("Live Run Status")
    render_run_status("Training")
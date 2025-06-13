import streamlit as st
import pandas as pd
import threading
import time
import os
import shutil
from pathlib import Path

from utils.database_manager import DatabaseManager
from utils.global_state import GLOBAL_APP_STATE, APP_LOCK
from engine.inference_controller import InferenceController
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer
from config import REPORTS_DIR, DB_PATH

st.set_page_config(page_title="Inference", page_icon="🔮", layout="wide")

def get_trained_models():
    if not REPORTS_DIR.is_dir(): return []
    return [d.name for d in REPORTS_DIR.iterdir() if d.is_dir() and (d / 'final_model').exists()]
def format_time(seconds):
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0: return "Calculating..."
    return time.strftime('%H:%M:%S', time.gmtime(seconds))
def reset_global_state():
    GLOBAL_APP_STATE.update({"is_task_running": False, "task_type": None, "log_buffer": [], "latest_progress": {}, "stop_requested": False, "result_buffer": None, "error": None, "done": False})
def callback_handler(data):
    if GLOBAL_APP_STATE["stop_requested"]: return "STOP"
    GLOBAL_APP_STATE["latest_progress"] = data
    if "log" in data: GLOBAL_APP_STATE["log_buffer"].append(data["log"])
    if "error" in data: GLOBAL_APP_STATE["error"] = data["error"]
    return "CONTINUE"
def run_batch_inference_in_thread(selected_run_id, uploaded_file, batch_size):
    controller, db_manager, visualizer = None, None, None; run_id = None
    with APP_LOCK:
        if GLOBAL_APP_STATE.get("is_task_running"): GLOBAL_APP_STATE["error"] = "Another task is already running."; return
        reset_global_state(); GLOBAL_APP_STATE["is_task_running"] = True; GLOBAL_APP_STATE["task_type"] = "Inference"
    try:
        db_manager = DatabaseManager(DB_PATH)
        run_id = db_manager.create_new_run('Inference', f"Inference on {selected_run_id[:8]}...", uploaded_file.name, {"inference_batch_size": batch_size})
        if not run_id: raise RuntimeError("Failed to create inference run in DB.")
        report_dir = REPORTS_DIR / run_id; report_dir.mkdir(exist_ok=True)
        visualizer = LogVisualizer(plot_dir=report_dir); monitor = ResourceMonitor(); monitor.start()
        controller = InferenceController(selected_run_id, callback=callback_handler)
        temp_output_path, perf_metrics = controller.predict_batch(uploaded_file, internal_batch_size=batch_size)
        final_status = 'FAILED'
        if temp_output_path and perf_metrics:
            final_output_path = report_dir / f"inference_results_{run_id}.csv"; shutil.move(temp_output_path, final_output_path)
            GLOBAL_APP_STATE["result_buffer"] = (str(final_output_path), perf_metrics); db_manager.save_performance_metrics(run_id, perf_metrics); final_status = 'COMPLETED'
        elif GLOBAL_APP_STATE["stop_requested"]: final_status = 'ABORTED'
        resource_metrics = monitor.stop(); visualizer.plot_resource_usage(resource_metrics); db_manager.save_resource_metrics(run_id, resource_metrics)
        report_path_for_db = str(report_dir) if final_status == 'COMPLETED' else None; db_manager.update_run_status(run_id, final_status, report_path_for_db)
    except Exception as e: print(f"Error during batch inference thread: {e}"); GLOBAL_APP_STATE["error"] = str(e)
    finally:
        if controller: controller.cleanup()
        if db_manager: db_manager.close()
        with APP_LOCK: GLOBAL_APP_STATE["is_task_running"] = False
        GLOBAL_APP_STATE["done"] = True

st.title("🔮 Perform Inference"); st.markdown("---")
is_any_task_running = GLOBAL_APP_STATE.get("is_task_running")
if is_any_task_running: st.warning(f"A '{GLOBAL_APP_STATE.get('task_type')}' task is currently running. All controls are disabled.")

col1, col2 = st.columns(2)
with col1:
    st.header("1. Select Model"); trained_models = get_trained_models()
    if not trained_models: st.warning("No trained models found."); st.stop()
    selected_run_id = st.selectbox("Select a Trained Model", options=trained_models, disabled=is_any_task_running)
    st.markdown("---")
    st.header("2. Choose Inference Mode")
    inference_mode = st.radio("Select inference type:", ["Single Prediction", "Batch Prediction"], horizontal=True, label_visibility="collapsed", disabled=is_any_task_running)
    if inference_mode == "Single Prediction":
        with st.container(border=True):
            st.subheader("Enter Log Sequence"); log_sequence_input = st.text_area("Use ' ;-; ' to separate log lines", height=150, disabled=is_any_task_running)
            if st.button("Predict Anomaly", use_container_width=True, disabled=is_any_task_running):
                if not selected_run_id or not log_sequence_input: st.warning("Please select a model and enter a log sequence.")
                else:
                    try:
                        with st.spinner("Performing prediction..."):
                            controller = InferenceController(selected_run_id); result = controller.predict_single(log_sequence_input); controller.cleanup()
                        if result['prediction'] == 'Anomalous': st.error(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
                        else: st.success(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
                    except Exception as e: st.error(f"An error occurred: {e}")
    elif inference_mode == "Batch Prediction":
        with st.container(border=True):
            st.subheader("Upload CSV File"); st.info("File must contain a 'Content' column. Predictions and confidences will be appended.")
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", disabled=is_any_task_running)
            batch_size = st.number_input("Inference Batch Size", min_value=1, max_value=256, value=64, help="Larger batches are faster but use more VRAM. Tune for your hardware.", disabled=is_any_task_running)
            if st.button("Start Batch Inference", use_container_width=True, disabled=is_any_task_running):
                if not selected_run_id or not uploaded_file: st.warning("Please select a model and upload a file.")
                else:
                    thread = threading.Thread(target=run_batch_inference_in_thread, args=(selected_run_id, uploaded_file, batch_size)); thread.start(); st.rerun()
    if GLOBAL_APP_STATE.get("result_buffer"):
        output_path, perf_metrics = GLOBAL_APP_STATE.get("result_buffer")
        st.success("Batch inference completed!");
        with open(output_path, "rb") as fp: st.download_button("Download Results CSV", fp, f"inference_results_{Path(output_path).stem}.csv", "text/csv")
        reset_global_state()
with col2:
    st.header("Live Batch Status")
    is_this_task_running = is_any_task_running and GLOBAL_APP_STATE.get("task_type") == "Inference"
    if is_this_task_running:
        progress_info = GLOBAL_APP_STATE.get("latest_progress", {}); progress = progress_info.get("progress", 0); rows = progress_info.get("rows_processed", 0); etc = progress_info.get("etc", 0)
        st.text(f"{progress:.1%}"); st.progress(progress); st.text(f"Rows Processed: {rows:,}"); st.text(f"ETC: {format_time(etc)}")
        if st.button("Stop Inference", use_container_width=True): GLOBAL_APP_STATE["stop_requested"] = True; st.warning("Stop request sent.")
        with st.expander("Show Live Logs", expanded=True): st.code('\n'.join(GLOBAL_APP_STATE.get("log_buffer", [])), language='log', height=500)
        if GLOBAL_APP_STATE.get("error"): st.error(f"An error occurred: {GLOBAL_APP_STATE['error']}")
        time.sleep(1); st.rerun()
    else: st.info("Status of a batch run will be displayed here.")
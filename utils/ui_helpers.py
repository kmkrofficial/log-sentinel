import streamlit as st
import time
import os
from .global_state import APP_LOCK, GLOBAL_APP_STATE

def format_time(seconds):
    """Formats seconds into a HH:MM:SS string."""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "Calculating..."
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def reset_global_state():
    """Resets the global state for a new run and cleans up temp files."""
    with APP_LOCK:
        GLOBAL_APP_STATE.update({
            "is_task_running": False,
            "task_type": None,
            "log_buffer": [],
            "latest_progress": {},
            "stop_requested": False,
            "result_buffer": None,
            "error": None,
            "done": False,
        })
        st.session_state.stop_button_clicked = False
        for key in list(st.session_state.keys()):
            if 'temp_file_path' in key:
                temp_path = st.session_state.pop(key, None)
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"Error cleaning up temp file {temp_path}: {e}")

def callback_handler(data):
    """Handles progress updates from a background thread."""
    if GLOBAL_APP_STATE.get("stop_requested", False):
        return "STOP"
    GLOBAL_APP_STATE["latest_progress"].update(data)
    if "log" in data:
        GLOBAL_APP_STATE["log_buffer"].append(data["log"])
    if "error" in data:
        GLOBAL_APP_STATE["error"] = data["error"]
    return "CONTINUE"

def render_run_status(task_type: str):
    """Renders the live status block for a running task."""
    is_this_task_running = (
        GLOBAL_APP_STATE.get("is_task_running")
        and GLOBAL_APP_STATE.get("task_type") == task_type
    )

    if is_this_task_running:
        progress_info = GLOBAL_APP_STATE.get("latest_progress", {})
        
        progress = progress_info.get("progress", 0)
        clamped_progress = min(progress, 1.0)
        progress_text = f"{clamped_progress:.1%}"

        st.text(f"Overall Progress: {progress_text}")
        st.progress(clamped_progress)
        
        if task_type == "Training":
            epoch = progress_info.get("epoch", "Starting...")
            loss = progress_info.get("loss", 0)
            st.text(f"Current Phase: {epoch}")
            st.text(f"Batch Loss: {loss:.4f}")

            # --- FIX: Display all three time metrics ---
            time_elapsed = progress_info.get("time_elapsed", 0)
            etc_phase = progress_info.get("etc_phase", 0)
            etc_overall = progress_info.get("etc_overall", 0)
            st.text(f"Time Elapsed:  {format_time(time_elapsed)}")
            st.text(f"Phase ETC:     {format_time(etc_phase)}")
            st.text(f"Overall ETC:   {format_time(etc_overall)}")

        else:
            rows = progress_info.get("rows_processed", 0)
            st.text(f"Rows Processed: {rows:,}")
            etc = progress_info.get("etc", 0)
            st.text(f"ETC: {format_time(etc)}")

        if 'stop_button_clicked' not in st.session_state:
            st.session_state.stop_button_clicked = False

        if st.button("🛑 Stop Run", type="secondary", use_container_width=True, disabled=st.session_state.stop_button_clicked):
            st.session_state.stop_button_clicked = True
            GLOBAL_APP_STATE["stop_requested"] = True
            st.warning("Stop request sent. The process will abort on the next step.")
            st.rerun()

        with st.expander("Show Live Logs", expanded=True):
            st.code(
                '\n'.join(GLOBAL_APP_STATE.get("log_buffer", [])),
                language='log',
                height=300
            )
        
        if error := GLOBAL_APP_STATE.get("error"):
            st.error(f"An error occurred: {error}")
        
        if not GLOBAL_APP_STATE.get("done", False):
            time.sleep(2)
            st.rerun()
    else:
        st.info("Status of a run will be displayed here once started.")
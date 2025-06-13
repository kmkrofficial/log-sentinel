import threading

# A true global lock shared by all sessions and threads in the app.
APP_LOCK = threading.Lock()

# A shared dictionary to hold the global state of the application.
# It tracks whatever long-running task is currently active.
GLOBAL_APP_STATE = {
    "is_task_running": False,
    "task_type": None, # "Training" or "Inference"
    "log_buffer": [],
    "latest_progress": {},
    "stop_requested": False,
    "result_buffer": None, # For inference results
    "error": None
}
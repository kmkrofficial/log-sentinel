import threading

APP_LOCK = threading.Lock()

GLOBAL_APP_STATE = {
    "is_task_running": False,
    "task_type": None,
    "log_buffer": [],
    "latest_progress": {},
    "stop_requested": False,
    "result_buffer": None,
    "error": None,
    "done": False,
}
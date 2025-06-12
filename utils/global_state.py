import threading

# A true global lock shared by all sessions and threads in the app.
APP_LOCK = threading.Lock()

# A shared dictionary to hold the global training status.
# The APP_LOCK must be used when accessing this dictionary.
TRAINING_STATUS = {
    "is_running": False,
    "current_run_id": None,
    "log_buffer": [],
    "latest_progress": {},
    "stop_requested": False
}
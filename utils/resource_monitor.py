import time
import threading
import psutil
import statistics

# Attempt to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    GPU_MONITORING_AVAILABLE = False

class ResourceMonitor:
    def __init__(self, interval=1.0, gpu_index=0):
        self.interval = interval
        self.process = psutil.Process()

        # Data storage
        self.timestamps = []
        self.cpu_usage = []
        self.ram_usage_mb = []
        self.gpu_util = []
        self.gpu_mem_mb = []

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)

        self.gpu_handle = None
        if GPU_MONITORING_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except pynvml.NVMLError:
                print(f"Warning: Could not get handle for GPU index {gpu_index}.")
                self.gpu_handle = None

    def _monitor_loop(self):
        """The internal loop that runs in a separate thread to collect metrics."""
        self.process.cpu_percent(interval=None) # Initialize CPU measurement
        
        while not self._stop_event.is_set():
            self.timestamps.append(time.time())
            
            # CPU and RAM
            self.cpu_usage.append(self.process.cpu_percent(interval=None))
            self.ram_usage_mb.append(self.process.memory_info().rss / (1024 * 1024))
            
            # GPU (if available)
            if self.gpu_handle:
                try:
                    gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_util.append(gpu_util_info.gpu)
                    self.gpu_mem_mb.append(gpu_mem_info.used / (1024 * 1024))
                except pynvml.NVMLError:
                    self.gpu_util.append(None)
                    self.gpu_mem_mb.append(None)

            time.sleep(self.interval)

    def start(self):
        """Starts the background monitoring thread."""
        print("Starting resource monitor...")
        self._thread.start()

    def stop(self):
        """Stops the monitoring thread and returns the collected metrics."""
        if self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
        print("Resource monitor stopped.")
        return self.get_metrics()
        
    def get_metrics(self):
        """
        Processes and returns the collected metrics as a dictionary.
        """
        if not self.timestamps:
            return {}

        # Align timestamps to be relative to the start
        start_time = self.timestamps[0]
        relative_timestamps = [ts - start_time for ts in self.timestamps]

        metrics = {
            "time_series": {
                "timestamps": relative_timestamps,
                "cpu_usage": self.cpu_usage,
                "ram_usage_mb": self.ram_usage_mb,
            },
            "summary": {
                "avg_cpu_usage": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                "peak_cpu_usage": max(self.cpu_usage) if self.cpu_usage else 0,
                "avg_ram_usage_mb": statistics.mean(self.ram_usage_mb) if self.ram_usage_mb else 0,
                "peak_ram_usage_mb": max(self.ram_usage_mb) if self.ram_usage_mb else 0,
            }
        }

        if self.gpu_handle:
            valid_gpu_util = [v for v in self.gpu_util if v is not None]
            valid_gpu_mem = [m for m in self.gpu_mem_mb if m is not None]
            metrics["time_series"]["gpu_util"] = self.gpu_util
            metrics["time_series"]["gpu_mem_mb"] = self.gpu_mem_mb
            metrics["summary"].update({
                "avg_gpu_util": statistics.mean(valid_gpu_util) if valid_gpu_util else 0,
                "peak_gpu_util": max(valid_gpu_util) if valid_gpu_util else 0,
                "avg_gpu_mem_mb": statistics.mean(valid_gpu_mem) if valid_gpu_mem else 0,
                "peak_gpu_mem_mb": max(valid_gpu_mem) if valid_gpu_mem else 0,
            })

        return metrics

    def __del__(self):
        if GPU_MONITORING_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass # Ignore errors on shutdown
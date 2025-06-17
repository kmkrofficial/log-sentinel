import time
import threading
import psutil
import statistics
import numpy as np

# We handle the library import and shutdown more robustly inside the class.
try:
    import pynvml
    PYNML_AVAILABLE = True
except ImportError:
    PYNML_AVAILABLE = False

class ResourceMonitor:
    def __init__(self, interval=1.0, gpu_index=0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.process = psutil.Process()
        self.cpu_count = psutil.cpu_count()
        self._stop_event = threading.Event()
        self._thread = None
        self.gpu_handle = None
        self.monitoring_active = False

        self.timestamps = []
        self.cpu_usage_percent = []
        self.ram_usage_gb = []
        
        self.gpu_util_percent = []
        self.gpu_mem_used_gb = []
        self.gpu_power_watts = []
        self.gpu_clock_mhz = []

    def _monitor_loop(self):
        # Initial call to establish a baseline for CPU percentage
        self.process.cpu_percent(interval=None)
        
        while not self._stop_event.is_set():
            self.timestamps.append(time.time())
            
            # CPU and RAM metrics
            self.cpu_usage_percent.append(self.process.cpu_percent(interval=None) / self.cpu_count)
            self.ram_usage_gb.append(self.process.memory_info().rss / (1024 ** 3))

            # GPU metrics, if available
            if self.gpu_handle:
                try:
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    
                    self.gpu_util_percent.append(util_rates.gpu)
                    self.gpu_mem_used_gb.append(mem_info.used / (1024 ** 3))
                    self.gpu_power_watts.append(pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0)
                    self.gpu_clock_mhz.append(pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_SM))
                except pynvml.NVMLError:
                    # Append None if reading fails mid-run
                    self.gpu_util_percent.append(None)
                    self.gpu_mem_used_gb.append(None)
                    self.gpu_power_watts.append(None)
                    self.gpu_clock_mhz.append(None)
            
            time.sleep(self.interval)

    def start(self):
        if self.monitoring_active:
            print("Resource monitor is already running.")
            return

        if PYNML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                print("pynvml initialized successfully. GPU monitoring enabled.")
            except pynvml.NVMLError as e:
                print(f"Warning: Could not initialize pynvml or get GPU handle: {e}. GPU monitoring will be disabled.")
                self.gpu_handle = None
        else:
            print("Warning: pynvml library not found. GPU monitoring is disabled.")

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.monitoring_active = True
        print("Resource monitor started.")

    def stop(self):
        if not self.monitoring_active:
            return {}

        self._stop_event.set()
        if self._thread:
            self._thread.join()
        
        # Get the metrics BEFORE shutting down pynvml. This is the fix.
        metrics = self.get_metrics()
        
        # Now, explicitly shut down pynvml to release the library
        if PYNML_AVAILABLE and self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
                print("pynvml shut down successfully.")
            except pynvml.NVMLError as e:
                print(f"Warning: Error during pynvml shutdown: {e}")

        self.monitoring_active = False
        print("Resource monitor stopped.")
        return metrics
        
    def get_metrics(self):
        if not self.timestamps:
            return {}
        
        start_time = self.timestamps[0]
        relative_timestamps = [ts - start_time for ts in self.timestamps]
        
        vmem = psutil.virtual_memory()
        total_system_ram_gb = vmem.total / (1024 ** 3)

        cpu_summary = {
            "avg_cpu_usage_percent": statistics.mean(self.cpu_usage_percent) if self.cpu_usage_percent else 0,
            "p95_cpu_usage_percent": np.percentile(self.cpu_usage_percent, 95) if self.cpu_usage_percent else 0,
            "utilized_cpu_time_sec": sum(self.process.cpu_times()),
        }

        ram_usage_percent = [(val / total_system_ram_gb) * 100 for val in self.ram_usage_gb]
        ram_summary = {
            "total_system_ram_gb": total_system_ram_gb,
            "avg_ram_usage_gb": statistics.mean(self.ram_usage_gb) if self.ram_usage_gb else 0,
            "p95_ram_usage_gb": np.percentile(self.ram_usage_gb, 95) if self.ram_usage_gb else 0,
            "avg_ram_usage_percent": statistics.mean(ram_usage_percent) if ram_usage_percent else 0,
            "p95_ram_usage_percent": np.percentile(ram_usage_percent, 95) if ram_usage_percent else 0
        }

        metrics = {
            "time_series": {
                "timestamps": relative_timestamps,
                "cpu_usage_percent": self.cpu_usage_percent,
                "ram_usage_gb": self.ram_usage_gb,
                "ram_usage_percent": ram_usage_percent,
            },
            "summary": {
                "cpu": cpu_summary,
                "ram": ram_summary
            }
        }

        if self.gpu_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                total_gpu_mem_gb = mem_info.total / (1024 ** 3)
            except pynvml.NVMLError:
                total_gpu_mem_gb = 0

            valid_gpu_util = [v for v in self.gpu_util_percent if v is not None]
            valid_gpu_mem_gb = [m for m in self.gpu_mem_used_gb if m is not None]
            valid_gpu_power = [p for p in self.gpu_power_watts if p is not None]
            valid_gpu_clock = [c for c in self.gpu_clock_mhz if c is not None]
            
            gpu_mem_usage_percent = [(val / total_gpu_mem_gb) * 100 for val in valid_gpu_mem_gb] if total_gpu_mem_gb > 0 else []

            gpu_summary = {
                "total_gpu_mem_gb": total_gpu_mem_gb,
                "avg_gpu_util_percent": statistics.mean(valid_gpu_util) if valid_gpu_util else 0,
                "p95_gpu_util_percent": np.percentile(valid_gpu_util, 95) if valid_gpu_util else 0,
                "avg_gpu_mem_gb": statistics.mean(valid_gpu_mem_gb) if valid_gpu_mem_gb else 0,
                "p95_gpu_mem_gb": np.percentile(valid_gpu_mem_gb, 95) if valid_gpu_mem_gb else 0,
                "avg_gpu_mem_percent": statistics.mean(gpu_mem_usage_percent) if gpu_mem_usage_percent else 0,
                "p95_gpu_mem_percent": np.percentile(gpu_mem_usage_percent, 95) if gpu_mem_usage_percent else 0,
                "avg_power_watts": statistics.mean(valid_gpu_power) if valid_gpu_power else 0,
                "p95_power_watts": np.percentile(valid_gpu_power, 95) if valid_gpu_power else 0,
                "avg_clock_mhz": statistics.mean(valid_gpu_clock) if valid_gpu_clock else 0,
            }

            ts_gpu_mem_percent = []
            if total_gpu_mem_gb > 0:
                ts_gpu_mem_percent = [(v / total_gpu_mem_gb) * 100 if v is not None else None for v in self.gpu_mem_used_gb]

            metrics["time_series"].update({
                "gpu_util_percent": self.gpu_util_percent,
                "gpu_mem_used_gb": self.gpu_mem_used_gb,
                "gpu_mem_usage_percent": ts_gpu_mem_percent,
                "gpu_power_watts": self.gpu_power_watts,
                "gpu_clock_mhz": self.gpu_clock_mhz,
            })
            metrics["summary"]["gpu"] = gpu_summary

        return metrics
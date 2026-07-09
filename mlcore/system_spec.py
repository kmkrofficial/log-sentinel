import platform
import psutil
import torch
import pynvml
import cpuinfo

def get_cpu_info():
    try:
        info = cpuinfo.get_cpu_info()
        return {
            "processor": info.get('brand_raw', 'N/A'),
            "arch": info.get('arch_string_raw', 'N/A'),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
        }
    except Exception:
        return {"error": "Could not retrieve CPU info."}

def get_ram_info():
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent,
        }
    except Exception:
        return {"error": "Could not retrieve RAM info."}

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"status": "CUDA not available."}
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram_gb = round(mem_info.total / (1024**3), 2)
        used_vram_gb = round(mem_info.used / (1024**3), 2)
        free_vram_gb = round(mem_info.free / (1024**3), 2)
        
        driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
        
        pynvml.nvmlShutdown()
        
        return {
            "device_name": device_name,
            "torch_cuda_version": torch.version.cuda,
            "driver_version": driver_version,
            "total_vram_gb": total_vram_gb,
            "used_vram_gb": used_vram_gb,
            "free_vram_gb": free_vram_gb,
        }
    except Exception as e:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return {"error": f"Could not retrieve GPU info via pynvml: {e}"}

def get_spec_dict():
    return {
        "platform": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "gpu": get_gpu_info(),
    }

if __name__ == "__main__":
    import json
    print(json.dumps(get_spec_dict(), indent=2))
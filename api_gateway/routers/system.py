import subprocess
import shutil
import psutil
from fastapi import APIRouter

router = APIRouter(prefix="/system", tags=["system"])

@router.get("/metrics")
def get_system_metrics():
    # 1. CPU & RAM Usage
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    
    # 2. Disk Space
    total, used, free = shutil.disk_usage("/")
    
    # 3. NVIDIA GPU VRAM & Utilization (via nvidia-smi command)
    gpu_name = "N/A"
    vram_used = 0
    vram_total = 0
    gpu_util = 0
    
    nvidia_smi = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if nvidia_smi:
        try:
            # Query GPU details
            out = subprocess.check_output([
                nvidia_smi, 
                "--query-gpu=name,memory.used,memory.total,utilization.gpu", 
                "--format=csv,noheader,nounits"
            ]).decode("utf-8").strip()
            
            parts = out.split(",")
            if len(parts) >= 4:
                gpu_name = parts[0].strip()
                vram_used = int(parts[1].strip())
                vram_total = int(parts[2].strip())
                gpu_util = int(parts[3].strip())
        except Exception:
            pass

    return {
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count(logical=True)
        },
        "ram": {
            "used_mb": ram.used // (1024 * 1024),
            "total_mb": ram.total // (1024 * 1024),
            "percent": ram.percent
        },
        "gpu": {
            "name": gpu_name,
            "vram_used_mb": vram_used,
            "vram_total_mb": vram_total,
            "percent": gpu_util,
            "vram_percent": round((vram_used / vram_total) * 100, 1) if vram_total > 0 else 0
        },
        "disk": {
            "percent": round((used / total) * 100, 1),
            "free_gb": free // (1024 * 1024 * 1024)
        }
    }

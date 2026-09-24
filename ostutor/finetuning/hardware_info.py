"""
Hardware Inspection Utility for OSTutorLLM.

Detects system specifications including CPU architecture, core count, RAM,
Apple Silicon / MPS availability, and CUDA / NVIDIA GPU availability.
"""

import os
import platform
import sys
import psutil
import torch


def get_hardware_info() -> dict:
    """
    Inspect local system hardware and accelerator availability.

    Returns:
        dict containing system metadata and device capabilities.
    """
    system_os = platform.system()
    arch = platform.machine()
    cpu_count = os.cpu_count() or 1
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    mps_available = False
    cuda_available = False
    gpu_name = None
    vram_gb = None
    accelerator_type = "CPU"

    # Check Apple Silicon / MPS
    if system_os == "Darwin" and (arch in ["arm64", "aarch64"]):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_available = True
            accelerator_type = "Apple Silicon (MPS)"

    # Check NVIDIA CUDA
    if torch.cuda.is_available():
        cuda_available = True
        accelerator_type = "NVIDIA CUDA"
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 2)

    return {
        "platform": system_os,
        "architecture": arch,
        "cpu_count": cpu_count,
        "ram_gb": total_ram_gb,
        "accelerator": accelerator_type,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def print_hardware_summary() -> None:
    """Print human-readable summary of system hardware."""
    info = get_hardware_info()
    print("==========================================")
    print(" OSTutorLLM Hardware Information ")
    print("==========================================")
    print(f"Platform    : {info['platform']}")
    print(f"Architecture: {info['architecture']}")
    print(f"CPU Cores   : {info['cpu_count']}")
    print(f"RAM         : {info['ram_gb']} GB")
    print(f"Accelerator : {info['accelerator']}")
    print(f"MPS Available: {'YES' if info['mps_available'] else 'NO'}")
    print(f"CUDA Available: {'YES' if info['cuda_available'] else 'NO'}")
    if info["gpu_name"]:
        print(f"GPU Name    : {info['gpu_name']}")
        print(f"VRAM        : {info['vram_gb']} GB")
    print(f"PyTorch Ver : {info['torch_version']}")
    print("==========================================")


if __name__ == "__main__":
    print_hardware_summary()

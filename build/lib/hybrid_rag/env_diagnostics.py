import os
import sys

def check_hardware_and_env(config_data: dict) -> dict:
    diagnostics = {
        "python_version": sys.version.split(' ')[0],
        "compute_device": "CPU",
        "missing_api_keys": []
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            diagnostics["compute_device"] = f"CUDA (Count: {torch.cuda.device_count()})"
        elif torch.backends.mps.is_available():
            diagnostics["compute_device"] = "MPS (Apple Silicon)"
    except ImportError:
        diagnostics["compute_device"] = "CPU (Torch not detected)"

    if config_data.get("mode") == "online":
        env_var = config_data.get("online", {}).get("api_key_env_var")
        if env_var and env_var not in os.environ:
            diagnostics["missing_api_keys"].append(env_var)

    return diagnostics
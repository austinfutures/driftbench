"""Small helpers: seeding, device selection, environment info, JSON I/O."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(pref: str = "auto"):
    import torch

    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _version(mod_name: str):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def environment_info(device=None) -> dict:
    """Everything a reader needs to know about where a number came from."""
    import torch

    info = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "mujoco": _version("mujoco"),
        "gymnasium": _version("gymnasium"),
        "ray": _version("ray"),
        "device": str(device) if device is not None else None,
        "device_type": None,
        "gpu_name": None,
    }
    if device is not None:
        dev = torch.device(device)
        info["device_type"] = "GPU" if dev.type in ("cuda", "mps") else "CPU"
        if dev.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(dev)
        elif dev.type == "mps":
            info["gpu_name"] = "Apple MPS"
    return info


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

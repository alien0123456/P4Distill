"""P4 export helpers for distillation models."""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np


def _quantize_tensor(tensor, mode: str, scale: Optional[float]):
    arr = tensor.detach().cpu().numpy()
    if mode == "none":
        return arr.tolist(), None

    max_abs = float(np.max(np.abs(arr))) if arr.size else 1.0
    if max_abs == 0.0:
        max_abs = 1.0

    if mode == "int8":
        qmax = 127
    elif mode == "int16":
        qmax = 32767
    else:
        raise ValueError(f"Unknown quantize mode: {mode}")

    use_scale = scale if scale is not None else max_abs / qmax
    if use_scale == 0.0:
        use_scale = 1.0

    q = np.clip(np.round(arr / use_scale), -qmax - 1, qmax).astype(int)
    return q.tolist(), float(use_scale)


def export_model_state_dict(
    state_dict,
    out_path: str,
    quantize: str = "int8",
    scale: Optional[float] = None,
    metadata: Optional[Dict] = None,
) -> None:
    payload = {
        "metadata": metadata or {},
        "quantize": quantize,
        "parameters": {},
    }

    for name, tensor in state_dict.items():
        values, used_scale = _quantize_tensor(tensor, quantize, scale)
        payload["parameters"][name] = {
            "shape": list(tensor.shape),
            "scale": used_scale,
            "values": values,
        }

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

"""PyTorch accelerator selection with Apple Metal support."""

from __future__ import annotations

import torch

SUPPORTED_DEVICES = {"auto", "cpu", "mps", "cuda"}


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "mps" and not torch.backends.mps.is_available():
        return False
    if device == "cuda" and not torch.cuda.is_available():
        return False
    try:
        torch.ones(1, device=device).sum().item()
    except (RuntimeError, NotImplementedError):
        return False
    return True


def resolve_device(requested: str) -> str:
    """Resolve auto to MPS, CUDA, or CPU and validate explicit accelerators."""

    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported PyTorch device: {requested}")
    if requested == "auto":
        for candidate in ("mps", "cuda", "cpu"):
            if _device_available(candidate):
                return candidate
        return "cpu"
    if not _device_available(requested):
        raise RuntimeError(f"Requested PyTorch device is unavailable: {requested}")
    return requested

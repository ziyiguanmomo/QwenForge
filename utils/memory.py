from __future__ import annotations

import torch


def cuda_available(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def synchronize(device: str) -> None:
    if cuda_available(device):
        torch.cuda.synchronize()


def clear_cuda(device: str) -> None:
    if cuda_available(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def reset_peak(device: str) -> None:
    if cuda_available(device):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def memory_snapshot(device: str) -> dict[str, float]:
    if not cuda_available(device):
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
    }

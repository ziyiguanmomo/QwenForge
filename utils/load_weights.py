from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def resolve_model_dir(model_path: str) -> Path:
    """Resolve a local model directory or a Hugging Face cache entry."""
    candidate = Path(model_path)
    if candidate.exists():
        return candidate

    cache_env = os.environ.get("HF_HUB_CACHE")
    if cache_env:
        cache_root = Path(cache_env)
    elif os.environ.get("HF_HOME"):
        cache_root = Path(os.environ["HF_HOME"]) / "hub"
    else:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    if "/" not in model_path:
        raise FileNotFoundError(f"Model path not found: {model_path}")

    namespace, name = model_path.split("/", maxsplit=1)
    cache_model_dir = cache_root / f"models--{namespace}--{name}"
    snapshots_dir = cache_model_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Model cache not found: {cache_model_dir}")

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under {snapshots_dir}")
    return snapshots[-1]


def load_config(model_path: str) -> dict[str, Any]:
    model_dir = resolve_model_dir(model_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def safetensor_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        return [model_dir / name for name in names]

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    return files


def load_state_dict(model_path: str, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load raw safetensors weights into a plain state_dict."""
    model_dir = resolve_model_dir(model_path)
    state_dict: dict[str, torch.Tensor] = {}
    for path in safetensor_files(model_dir):
        shard = load_file(str(path), device=str(device))
        overlap = set(state_dict).intersection(shard)
        if overlap:
            joined = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Duplicate tensor names while loading {path.name}: {joined}")
        state_dict.update(shard)
    return state_dict


def load_config_and_state_dict(
    model_path: str,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Convenience helper that returns config and raw weights only."""
    return load_config(model_path), load_state_dict(model_path, device=device)

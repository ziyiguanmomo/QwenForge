#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import os
import random
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import load_jsonl, write_json, write_text
from utils.memory import clear_cuda, memory_snapshot, reset_peak, synchronize
from utils.metrics import (
    batch_scaling_factor,
    compute_final_score,
    keyword_score,
    mean,
    output_token_count,
    required_substring_score,
    safe_div,
    suite_score_view,
    substring_score,
)


DEFAULT_MODEL = "/data/course_env/models/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "results/final_eval"
SUITE_LONG = "long_context"
SUITE_DECODE = "decode_throughput"
SUITE_TTFT = "ttft_prefill"
SUITE_MIXED = "mixed_serving"
SUITE_SERVING = "serving_schedule"
SUITE_CACHE_STRESS = "decode_cache_stress"
ALL_SUITES = [SUITE_LONG, SUITE_DECODE, SUITE_TTFT, SUITE_SERVING, SUITE_MIXED, SUITE_CACHE_STRESS]


def set_cache_env() -> None:
    user = os.environ.get("USER", "student")
    project_root = Path(__file__).resolve().parents[1]
    default_cache_root = Path(f"/data/{user}/cache")
    fallback_cache_root = project_root / ".cache"
    cache_root = Path(os.environ.get("INFERENCE_OPT_CACHE_ROOT", default_cache_root))

    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_root = fallback_cache_root
        cache_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets"))
    os.environ.setdefault("NLTK_DATA", str(cache_root / "nltk"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("TMPDIR", str(cache_root / "tmp"))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))

    for key in [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "NLTK_DATA",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "PIP_CACHE_DIR",
    ]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["student"], default="student")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--device", default=os.environ.get("HF_DEVICE", "cuda"))
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--suites", default=",".join(ALL_SUITES))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--worker-timeout-s", type=float, default=0.0)
    parser.add_argument("--baseline-summary", default="data/public_baseline_summary.json")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--long-context-data", default="data/public_long_context.jsonl")
    parser.add_argument("--decode-throughput-data", default="data/public_decode_throughput.jsonl")
    parser.add_argument("--ttft-data", default="data/public_ttft_prefill.jsonl")
    parser.add_argument("--serving-schedule-data", default="data/public_serving_schedule.jsonl")
    parser.add_argument("--mixed-serving-data", default="data/public_mixed_serving.jsonl")
    parser.add_argument("--decode-cache-stress-data", default="data/public_decode_cache_stress.jsonl")

    parser.add_argument("--max-new-tokens-long", type=int, default=96)
    parser.add_argument("--max-new-tokens-decode", type=int, default=128)
    parser.add_argument("--max-new-tokens-ttft", type=int, default=1)
    parser.add_argument("--max-new-tokens-serving", type=int, default=96)
    parser.add_argument("--max-new-tokens-mixed", type=int, default=64)
    parser.add_argument("--max-new-tokens-cache-stress", default="128,256,512")
    parser.add_argument("--decode-batch-sizes", default="1,2,4")
    parser.add_argument("--ttft-batch-sizes", default="1")
    parser.add_argument(
        "--serving-fallback-batch-size",
        type=int,
        default=4,
        help="Fallback generate() batch size when serve_requests() is not implemented.",
    )
    parser.add_argument(
        "--serving-batch-sizes",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mixed-batch-sizes", default="1,2")
    parser.add_argument("--cache-stress-batch-sizes", default="2,4")
    parser.add_argument("--verbose-batches", action="store_true", help="Print every benchmark batch.")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Global warmup generate calls before measured suites.")
    parser.add_argument("--batch-warmup-iters", type=int, default=0, help="Optional per-batch warmup calls before timed repeats.")
    parser.add_argument("--timed-repeats", type=int, default=1, help="Measured repeats per batch; latency uses the median repeat.")
    parser.add_argument(
        "--suite-isolation",
        choices=["process", "shared"],
        default="process",
        help="Run each suite in a fresh worker process, or reuse one engine for all suites.",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip static validation before controller launches worker.")
    parser.add_argument(
        "--allow-stale-baseline",
        action="store_true",
        help="Allow scoring with a baseline summary that has no matching data fingerprints. Debug only.",
    )
    args = parser.parse_args()
    if args.serving_batch_sizes:
        legacy_values = [int(item) for item in str(args.serving_batch_sizes).split(",") if item.strip()]
        if legacy_values:
            args.serving_fallback_batch_size = max(max(legacy_values), 1)
    return args


def set_random_seed(seed: int) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def selected_suites(value: str) -> list[str]:
    suites = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(suites) - set(ALL_SUITES))
    if unknown:
        raise ValueError(f"Unknown suite(s): {', '.join(unknown)}")
    return suites


def suite_data_path(args: argparse.Namespace, suite: str) -> str:
    if suite == SUITE_LONG:
        return resolve_project_path(args.long_context_data)
    if suite == SUITE_DECODE:
        return resolve_project_path(args.decode_throughput_data)
    if suite == SUITE_TTFT:
        return resolve_project_path(args.ttft_data)
    if suite == SUITE_SERVING:
        return resolve_project_path(args.serving_schedule_data)
    if suite == SUITE_MIXED:
        return resolve_project_path(args.mixed_serving_data)
    if suite == SUITE_CACHE_STRESS:
        return resolve_project_path(args.decode_cache_stress_data)
    raise ValueError(suite)


def resolve_project_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT / candidate)


def suite_batch_sizes(args: argparse.Namespace, suite: str) -> list[int]:
    if suite == SUITE_LONG:
        return [1]
    if suite == SUITE_DECODE:
        return parse_csv_ints(args.decode_batch_sizes)
    if suite == SUITE_TTFT:
        return parse_csv_ints(args.ttft_batch_sizes)
    if suite == SUITE_SERVING:
        return [max(int(args.serving_fallback_batch_size), 1)]
    if suite == SUITE_MIXED:
        return parse_csv_ints(args.mixed_batch_sizes)
    if suite == SUITE_CACHE_STRESS:
        return parse_csv_ints(args.cache_stress_batch_sizes)
    raise ValueError(suite)


def suite_max_new_token_values(args: argparse.Namespace, suite: str) -> list[int]:
    if suite == SUITE_LONG:
        return [args.max_new_tokens_long]
    if suite == SUITE_DECODE:
        return [args.max_new_tokens_decode]
    if suite == SUITE_TTFT:
        return [args.max_new_tokens_ttft]
    if suite == SUITE_SERVING:
        return [args.max_new_tokens_serving]
    if suite == SUITE_MIXED:
        return [args.max_new_tokens_mixed]
    if suite == SUITE_CACHE_STRESS:
        return parse_csv_ints(args.max_new_tokens_cache_stress)
    raise ValueError(suite)


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    line_count = 0
    with source.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
            line_count += block.count(b"\n")
    return {
        "path": str(source),
        "name": source.name,
        "bytes": source.stat().st_size,
        "lines": line_count,
        "sha256": digest.hexdigest(),
    }


def collect_data_fingerprints(args: argparse.Namespace) -> dict[str, Any]:
    fingerprints: dict[str, Any] = {}
    for suite in selected_suites(args.suites):
        fingerprints[suite] = file_fingerprint(suite_data_path(args, suite))
    return fingerprints


def collect_runtime_env(args: argparse.Namespace) -> dict[str, Any]:
    gpu_name = "N/A"
    gpu_count = 0
    cuda_device_index = "N/A"
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        gpu_count = torch.cuda.device_count()
        cuda_device_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(cuda_device_index)
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "cuda_device_index": cuda_device_index,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "",
        "seed": int(args.seed),
    }


def collect_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": bool(args.local_files_only),
        "suites": args.suites,
        "limit": args.limit,
        "seed": int(args.seed),
        "decode_batch_sizes": args.decode_batch_sizes,
        "ttft_batch_sizes": args.ttft_batch_sizes,
        "serving_mode": "request_stream",
        "serving_fallback_batch_size": int(args.serving_fallback_batch_size),
        "mixed_batch_sizes": args.mixed_batch_sizes,
        "cache_stress_batch_sizes": args.cache_stress_batch_sizes,
        "max_new_tokens_long": args.max_new_tokens_long,
        "max_new_tokens_decode": args.max_new_tokens_decode,
        "max_new_tokens_ttft": args.max_new_tokens_ttft,
        "max_new_tokens_serving": args.max_new_tokens_serving,
        "max_new_tokens_mixed": args.max_new_tokens_mixed,
        "max_new_tokens_cache_stress": args.max_new_tokens_cache_stress,
        "warmup_iters": int(args.warmup_iters),
        "batch_warmup_iters": int(args.batch_warmup_iters),
        "timed_repeats": int(args.timed_repeats),
        "suite_isolation": args.suite_isolation,
    }


def case_id(row: dict[str, Any], index: int) -> str:
    if row.get("case_id"):
        return str(row["case_id"])
    task = str(row.get("task", "case"))
    sample_index = row.get("sample_index", row.get("subset_index", index))
    return f"{task}_{sample_index}"


def case_references(row: dict[str, Any], suite: str) -> list[str]:
    if suite == SUITE_LONG:
        return [str(item) for item in row.get("outputs", row.get("required_substrings", []))]
    if row.get("strict_required_substrings"):
        return [str(item) for item in row.get("required_substrings", [])]
    return [str(item) for item in row.get("required_exact", [])]


def case_keywords(row: dict[str, Any]) -> list[str]:
    return [str(item) for item in row.get("required_keywords", [])]


def case_keyword_threshold(row: dict[str, Any]) -> float:
    if "required_keyword_threshold" in row:
        return float(row["required_keyword_threshold"])
    if "required_keyword_min" in row:
        keywords = case_keywords(row)
        return safe_div(float(row["required_keyword_min"]), max(len(keywords), 1), default=1.0)
    return 0.34


def normalize_for_copy_guard(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def copied_prompt_prefix_tokens(tokenizer, prompt: str, generated_text: str) -> int:
    """Estimate how many leading output tokens are copied verbatim from prompt.

    The benchmark cannot see a student's internal decode loop. This guard makes
    performance metrics less sensitive to implementations that prepend long,
    prompt-derived fixed text and then decode only a tiny tail.
    """
    generated = (generated_text or "").strip()
    if not generated:
        return 0

    prompt_norm = normalize_for_copy_guard(prompt)
    if not prompt_norm:
        return 0

    token_ids = tokenizer(generated, add_special_tokens=False).get("input_ids", [])
    if not token_ids:
        return 0

    best = 0
    # Limit the scan so the guard is cheap even for long outputs.
    max_scan = min(len(token_ids), 160)
    for length in range(1, max_scan + 1):
        prefix = tokenizer.decode(
            token_ids[:length],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prefix_norm = normalize_for_copy_guard(prefix)
        if prefix_norm and prefix_norm in prompt_norm:
            best = length
        elif length - best > 16:
            break
    return best


def chunks(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start // batch_size, items[start : start + batch_size]


def load_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_engine(args: argparse.Namespace):
    module = importlib.import_module("student_engine")
    engine_cls = getattr(module, "StudentEngine")
    kwargs = {
        "model_path": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
        "seed": args.seed,
    }
    signature = inspect.signature(engine_cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        filtered = kwargs
    else:
        filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return engine_cls(**filtered)


def make_failure_rows(
    suite: str,
    batch_size: int,
    batch_id: str,
    batch_rows: list[dict[str, Any]],
    tokenizer,
    max_new_tokens: int,
    latency_s: float,
    init_memory: dict[str, float],
    peak: dict[str, float],
    error_type: str,
    error_message: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for offset, row in enumerate(batch_rows):
        prompt = str(row["prompt"])
        output.append(
            build_result_row(
                suite=suite,
                batch_size=batch_size,
                batch_id=batch_id,
                row=row,
                row_index=offset,
                tokenizer=tokenizer,
                prompt=prompt,
                generated_text="",
                max_new_tokens=max_new_tokens,
                latency_s=latency_s,
                init_memory=init_memory,
                peak=peak,
                error_type=error_type,
                error_message=error_message,
                valid_output=False,
                diversity_ok=False,
            )
        )
    return output


def build_result_row(
    suite: str,
    batch_size: int,
    batch_id: str,
    row: dict[str, Any],
    row_index: int,
    tokenizer,
    prompt: str,
    generated_text: str,
    max_new_tokens: int,
    latency_s: float,
    init_memory: dict[str, float],
    peak: dict[str, float],
    error_type: str,
    error_message: str,
    valid_output: bool,
    diversity_ok: bool,
) -> dict[str, Any]:
    prompt_tokens = output_token_count(tokenizer, prompt)
    generated_tokens = output_token_count(tokenizer, generated_text)
    copied_prefix_tokens = (
        0 if suite == SUITE_TTFT else copied_prompt_prefix_tokens(tokenizer, prompt, generated_text)
    )
    prompt_copy_ratio = safe_div(copied_prefix_tokens, max(generated_tokens, 1), default=0.0)
    scored_generated_tokens = max(0, generated_tokens - copied_prefix_tokens)
    raw_effective_generated_tokens = min(generated_tokens, max_new_tokens)
    effective_generated_tokens = (
        raw_effective_generated_tokens if suite == SUITE_TTFT else min(scored_generated_tokens, max_new_tokens)
    )
    generated_ratio = safe_div(effective_generated_tokens, max_new_tokens, default=0.0)
    raw_generated_ratio = safe_div(raw_effective_generated_tokens, max_new_tokens, default=0.0)
    too_long = generated_tokens > int(max_new_tokens * 1.25) + 8
    references = case_references(row, suite)
    keywords = case_keywords(row)
    keyword_threshold = case_keyword_threshold(row)

    if suite == SUITE_LONG:
        score = substring_score(generated_text, references)
        required_score = score
        relevance_score = 1.0
        correct = score >= 1.0
        non_empty = bool(generated_text.strip())
        success = error_type == "" and non_empty and not too_long
        valid = success
    elif suite == SUITE_TTFT:
        required_score = 1.0
        relevance_score = 1.0
        non_empty = bool(generated_text.strip())
        length_ok = raw_effective_generated_tokens >= 1
        valid = error_type == "" and valid_output and non_empty and length_ok and not too_long
        score = 1.0 if valid else 0.0
        correct = valid
        success = valid
    else:
        required_score = required_substring_score(generated_text, references)
        relevance_score = keyword_score(generated_text, keywords)
        length_ok = generated_ratio >= 0.80
        non_empty = bool(generated_text.strip())
        valid = (
            error_type == ""
            and valid_output
            and diversity_ok
            and non_empty
            and length_ok
            and not too_long
            and required_score >= 1.0
            and relevance_score >= keyword_threshold
        )
        score = 1.0 if valid else 0.0
        correct = valid
        success = valid

    peak_extra_allocated = max(0.0, peak["peak_allocated_mb"] - init_memory["allocated_mb"])
    peak_extra_reserved = max(0.0, peak["peak_reserved_mb"] - init_memory["reserved_mb"])

    return {
        "suite": suite,
        "case_id": case_id(row, row_index),
        "task": row.get("task", suite),
        "workload_type": row.get("workload_type", ""),
        "prompt_length_bucket": row.get("prompt_length_bucket", ""),
        "shared_prefix_id": row.get("shared_prefix_id", ""),
        "serving_interface": row.get("serving_interface", ""),
        "batch_size": batch_size,
        "batch_id": batch_id,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generated_tokens,
        "scored_generated_tokens": scored_generated_tokens,
        "effective_generated_tokens": effective_generated_tokens,
        "raw_effective_generated_tokens": raw_effective_generated_tokens,
        "generated_token_ratio": generated_ratio,
        "raw_generated_token_ratio": raw_generated_ratio,
        "copied_prompt_prefix_tokens": copied_prefix_tokens,
        "copied_prompt_prefix_ratio": prompt_copy_ratio,
        "score": score,
        "correct": correct,
        "required_score": required_score,
        "keyword_score": relevance_score,
        "keyword_threshold": keyword_threshold,
        "valid_output": valid,
        "success": success,
        "latency_s": latency_s,
        "tokens_per_s": safe_div(effective_generated_tokens if valid else 0.0, latency_s),
        "init_allocated_mb": init_memory["allocated_mb"],
        "init_reserved_mb": init_memory["reserved_mb"],
        "peak_allocated_mb": peak["peak_allocated_mb"],
        "peak_reserved_mb": peak["peak_reserved_mb"],
        "peak_extra_allocated_mb": peak_extra_allocated,
        "peak_extra_reserved_mb": peak_extra_reserved,
        "too_long": too_long,
        "diversity_ok": diversity_ok,
        "error_type": error_type,
        "error_message": error_message[:240],
        "required_substrings": "|".join(references),
        "required_keywords": "|".join(keywords),
        "nonce": row.get("nonce", ""),
        "generated_text": generated_text.replace("\n", "\\n")[:1000],
    }


def build_serving_requests(
    batch_id: str,
    batch_rows: list[dict[str, Any]],
    max_new_tokens: int,
    stream_size: int,
    fallback_batch_size: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for offset, row in enumerate(batch_rows):
        request_id = str(row.get("request_id") or f"{batch_id}:{case_id(row, offset)}")
        requests.append(
            {
                "request_id": request_id,
                "prompt": str(row["prompt"]),
                "max_new_tokens": int(row.get("max_new_tokens", max_new_tokens)),
                "arrival_time_ms": float(row.get("arrival_time_ms", offset * 10.0)),
                "priority": int(row.get("priority", 0)),
                "group_id": str(row.get("shared_prefix_id") or row.get("group_id") or ""),
                "workload_type": str(row.get("workload_type", "")),
                "prompt_length_bucket": str(row.get("prompt_length_bucket", "")),
                "benchmark_mode": "request_stream",
                "decode_mode": "fixed_step_ignore_eos",
                "ignore_eos": True,
                "stream_size": int(stream_size),
                "fallback_batch_size": int(fallback_batch_size),
            }
        )
    return requests


def text_from_serving_item(item: Any) -> str:
    if isinstance(item, dict):
        for key in ["generated_text", "text", "output", "response"]:
            if key in item:
                return "" if item[key] is None else str(item[key])
        if "outputs" in item:
            return text_from_serving_item(item["outputs"])
    if isinstance(item, list) and item:
        return text_from_serving_item(item[0])
    return "" if item is None else str(item)


def normalize_serving_outputs(outputs: Any, requests: list[dict[str, Any]]) -> list[str]:
    request_ids = [str(item["request_id"]) for item in requests]
    if isinstance(outputs, dict):
        normalized: list[str] = []
        for offset, request_id in enumerate(request_ids):
            value = outputs.get(request_id, outputs.get(offset, outputs.get(str(offset))))
            if value is None:
                raise ValueError(f"serve_requests() missing output for request_id={request_id!r}")
            normalized.append(text_from_serving_item(value))
        return normalized

    if not isinstance(outputs, list):
        raise TypeError(f"serve_requests() must return list or dict, got {type(outputs).__name__}")
    if len(outputs) != len(requests):
        raise ValueError(f"serve_requests() returned {len(outputs)} outputs for {len(requests)} requests")

    if outputs and all(isinstance(item, dict) and "request_id" in item for item in outputs):
        by_id = {str(item["request_id"]): item for item in outputs}
        return [text_from_serving_item(by_id[request_id]) for request_id in request_ids]

    return [text_from_serving_item(item) for item in outputs]


def call_serving_interface(engine, requests: list[dict[str, Any]], batch_size: int | None) -> Any:
    serve_requests = getattr(engine, "serve_requests", None)
    if not callable(serve_requests):
        return None

    signature = inspect.signature(serve_requests)
    accepts_batch_size = (
        "batch_size" in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )
    if accepts_batch_size:
        return serve_requests(requests, batch_size=batch_size)
    return serve_requests(requests)


def run_batch(
    engine,
    tokenizer,
    args: argparse.Namespace,
    suite: str,
    batch_size: int,
    batch_id: str,
    batch_rows: list[dict[str, Any]],
    max_new_tokens: int,
    init_memory: dict[str, float],
    generate_batch_size: int | None = None,
    serving_stream: bool = False,
) -> list[dict[str, Any]]:
    prompts = [str(row["prompt"]) for row in batch_rows]
    reset_peak(args.device)
    start = time.perf_counter()
    outputs: list[str] | None = None
    error_type = ""
    error_message = ""

    try:
        serving_outputs = None
        serving_interface = "generate"
        if suite == SUITE_SERVING:
            fallback_batch_size = max(int(generate_batch_size or batch_size), 1)
            requests = build_serving_requests(
                batch_id=batch_id,
                batch_rows=batch_rows,
                max_new_tokens=max_new_tokens,
                stream_size=len(batch_rows),
                fallback_batch_size=fallback_batch_size,
            )
            serving_outputs = call_serving_interface(engine, requests, None if serving_stream else batch_size)

        if serving_outputs is not None:
            serving_interface = "serve_requests"
            outputs = normalize_serving_outputs(serving_outputs, requests)
        else:
            outputs = engine.generate(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=max(int(generate_batch_size or batch_size), 1),
                suite_name=None,
            )
            if not isinstance(outputs, list):
                raise TypeError(f"generate() must return list[str], got {type(outputs).__name__}")
            if len(outputs) != len(prompts):
                raise ValueError(f"generate() returned {len(outputs)} outputs for {len(prompts)} prompts")
            outputs = ["" if item is None else str(item) for item in outputs]
    except torch.cuda.OutOfMemoryError as exc:
        error_type = "cuda_oom"
        error_message = str(exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        traceback.print_exc()

    synchronize(args.device)
    latency_s = time.perf_counter() - start
    peak = memory_snapshot(args.device)

    if outputs is None:
        return make_failure_rows(
            suite=suite,
            batch_size=batch_size,
            batch_id=batch_id,
            batch_rows=batch_rows,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            latency_s=latency_s,
            init_memory=init_memory,
            peak=peak,
            error_type=error_type,
            error_message=error_message,
        )

    stripped = [item.strip() for item in outputs]
    diversity_ok = not (len(stripped) > 1 and len(set(stripped)) == 1)
    output_rows: list[dict[str, Any]] = []
    for offset, (row, generated_text) in enumerate(zip(batch_rows, outputs)):
        result_row = build_result_row(
            suite=suite,
            batch_size=batch_size,
            batch_id=batch_id,
            row=row,
            row_index=offset,
            tokenizer=tokenizer,
            prompt=prompts[offset],
            generated_text=generated_text,
            max_new_tokens=max_new_tokens,
            latency_s=latency_s,
            init_memory=init_memory,
            peak=peak,
            error_type=error_type,
            error_message=error_message,
            valid_output=True,
            diversity_ok=diversity_ok,
        )
        if suite == SUITE_SERVING:
            result_row["serving_interface"] = serving_interface
        output_rows.append(result_row)
    return output_rows


def run_measured_batch(
    engine,
    tokenizer,
    args: argparse.Namespace,
    suite: str,
    batch_size: int,
    batch_id: str,
    batch_rows: list[dict[str, Any]],
    max_new_tokens: int,
    init_memory: dict[str, float],
    generate_batch_size: int | None = None,
    serving_stream: bool = False,
) -> list[dict[str, Any]]:
    for warmup_index in range(max(int(args.batch_warmup_iters), 0)):
        run_batch(
            engine=engine,
            tokenizer=tokenizer,
            args=args,
            suite=suite,
            batch_size=batch_size,
            batch_id=f"{batch_id}_warmup{warmup_index}",
            batch_rows=batch_rows,
            max_new_tokens=max_new_tokens,
            init_memory=init_memory,
            generate_batch_size=generate_batch_size,
            serving_stream=serving_stream,
        )
        reset_peak(args.device)

    repeats = max(int(args.timed_repeats), 1)
    attempts: list[list[dict[str, Any]]] = []
    for repeat_index in range(repeats):
        attempt_rows = run_batch(
            engine=engine,
            tokenizer=tokenizer,
            args=args,
            suite=suite,
            batch_size=batch_size,
            batch_id=f"{batch_id}_r{repeat_index}",
            batch_rows=batch_rows,
            max_new_tokens=max_new_tokens,
            init_memory=init_memory,
            generate_batch_size=generate_batch_size,
            serving_stream=serving_stream,
        )
        attempts.append(attempt_rows)

    attempts.sort(key=lambda rows: float(rows[0]["latency_s"]) if rows else float("inf"))
    selected_index = len(attempts) // 2
    selected_rows = attempts[selected_index]
    peak_allocated = max(
        (float(row["peak_allocated_mb"]) for rows in attempts for row in rows),
        default=0.0,
    )
    peak_reserved = max(
        (float(row["peak_reserved_mb"]) for rows in attempts for row in rows),
        default=0.0,
    )
    for row in selected_rows:
        row["batch_id"] = batch_id
        row["timed_repeats"] = repeats
        row["selected_repeat"] = selected_index
        if peak_allocated > 0.0:
            row["peak_allocated_mb"] = peak_allocated
            row["peak_extra_allocated_mb"] = max(0.0, peak_allocated - float(row["init_allocated_mb"]))
        if peak_reserved > 0.0:
            row["peak_reserved_mb"] = peak_reserved
            row["peak_extra_reserved_mb"] = max(0.0, peak_reserved - float(row["init_reserved_mb"]))
    return selected_rows


def run_suite(
    engine,
    tokenizer,
    args: argparse.Namespace,
    suite: str,
    init_memory: dict[str, float],
) -> list[dict[str, Any]]:
    data_path = suite_data_path(args, suite)
    rows = load_jsonl(data_path, limit=args.limit)
    batch_sizes = suite_batch_sizes(args, suite)
    max_new_token_values = suite_max_new_token_values(args, suite)
    results: list[dict[str, Any]] = []

    if suite == SUITE_SERVING:
        fallback_batch_size = max(int(args.serving_fallback_batch_size), 1)
        print(
            f"\n[{suite}] data={data_path} requests={len(rows)} "
            f"mode=request_stream fallback_generate_batch_size={fallback_batch_size} "
            f"max_new_tokens={max_new_token_values}"
        )
        for max_new_tokens in max_new_token_values:
            batch_id = f"{suite}_stream{len(rows)}_tok{max_new_tokens}_0000"
            batch_results = run_measured_batch(
                engine=engine,
                tokenizer=tokenizer,
                args=args,
                suite=suite,
                batch_size=max(len(rows), 1),
                batch_id=batch_id,
                batch_rows=rows,
                max_new_tokens=max_new_tokens,
                init_memory=init_memory,
                generate_batch_size=fallback_batch_size,
                serving_stream=True,
            )
            results.extend(batch_results)
            ok = sum(1 for item in batch_results if item["success"])
            total = len(batch_results)
            latency = batch_results[0]["latency_s"] if batch_results else 0.0
            if args.verbose_batches:
                print(
                    f"  stream_requests={len(rows):<3} fallback_bs={fallback_batch_size:<2} "
                    f"max_new={max_new_tokens:<3} success={ok}/{total} latency={latency:.3f}s"
                )
        return results

    print(
        f"\n[{suite}] data={data_path} cases={len(rows)} "
        f"batch_sizes={batch_sizes} max_new_tokens={max_new_token_values}"
    )
    for batch_size in batch_sizes:
        for max_new_tokens in max_new_token_values:
            for batch_index, batch_rows in chunks(rows, batch_size):
                batch_id = f"{suite}_bs{batch_size}_tok{max_new_tokens}_{batch_index:04d}"
                batch_results = run_measured_batch(
                    engine=engine,
                    tokenizer=tokenizer,
                    args=args,
                    suite=suite,
                    batch_size=batch_size,
                    batch_id=batch_id,
                    batch_rows=batch_rows,
                    max_new_tokens=max_new_tokens,
                    init_memory=init_memory,
                )
                results.extend(batch_results)
                ok = sum(1 for item in batch_results if item["success"])
                total = len(batch_results)
                latency = batch_results[0]["latency_s"] if batch_results else 0.0
                if args.verbose_batches:
                    print(
                        f"  batch_size={batch_size:<2} max_new={max_new_tokens:<3} "
                        f"batch={batch_index:<3} success={ok}/{total} latency={latency:.3f}s"
                    )

    return results


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    batch_latency = {}
    for row in rows:
        batch_latency[row["batch_id"]] = float(row["latency_s"])
    total_latency = sum(batch_latency.values())
    runtime_rows = [row for row in rows if row["error_type"] == "" and not row.get("too_long", False)]
    total_effective_tokens = sum(float(row["effective_generated_tokens"]) for row in runtime_rows)
    total_raw_effective_tokens = sum(float(row.get("raw_effective_generated_tokens", row["effective_generated_tokens"])) for row in runtime_rows)
    latencies = [float(row["latency_s"]) for row in rows]
    latencies_sorted = sorted(latencies)
    p95_rank = (len(latencies_sorted) * 95 + 99) // 100
    p95_index = min(len(latencies_sorted) - 1, max(p95_rank - 1, 0))
    interface_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        interface = str(row.get("serving_interface", "")).strip()
        if interface:
            interface_counts[interface] += 1
    primary_interface = ""
    if interface_counts:
        primary_interface = max(interface_counts, key=lambda key: interface_counts[key])

    return {
        "n": len(rows),
        "success_rate": mean([1.0 if row["success"] else 0.0 for row in rows]),
        "runtime_success_rate": mean([1.0 if row["error_type"] == "" else 0.0 for row in rows]),
        "valid_output_rate": mean([1.0 if row["valid_output"] else 0.0 for row in rows]),
        "accuracy": mean([1.0 if row["correct"] else 0.0 for row in rows]),
        "avg_score": mean([float(row["score"]) for row in rows]),
        "avg_generated_ratio": mean([float(row["generated_token_ratio"]) for row in rows]),
        "tokens_per_s": safe_div(total_effective_tokens, total_latency, default=0.0),
        "requests_per_s": safe_div(len(rows), total_latency, default=0.0),
        "avg_latency_s": mean(latencies),
        "p95_latency_s": latencies_sorted[p95_index] if latencies_sorted else 0.0,
        "total_latency_s": total_latency,
        "generated_tokens": total_effective_tokens,
        "raw_generated_tokens": total_raw_effective_tokens,
        "avg_copied_prompt_prefix_tokens": mean([float(row.get("copied_prompt_prefix_tokens", 0.0)) for row in rows]),
        "avg_copied_prompt_prefix_ratio": mean([float(row.get("copied_prompt_prefix_ratio", 0.0)) for row in rows]),
        "init_allocated_mb": max(float(row["init_allocated_mb"]) for row in rows),
        "init_reserved_mb": max(float(row["init_reserved_mb"]) for row in rows),
        "peak_allocated_mb": max(float(row["peak_allocated_mb"]) for row in rows),
        "peak_reserved_mb": max(float(row["peak_reserved_mb"]) for row in rows),
        "peak_extra_allocated_mb": max(float(row["peak_extra_allocated_mb"]) for row in rows),
        "peak_extra_reserved_mb": max(float(row["peak_extra_reserved_mb"]) for row in rows),
        "oom_count": sum(1 for row in rows if row["error_type"] == "cuda_oom"),
        "serving_interface": primary_interface,
        "serve_requests_used_rate": mean(
            [1.0 if str(row.get("serving_interface", "")) == "serve_requests" else 0.0 for row in rows]
        ),
    }


def fit_memory_growth(points: list[dict[str, float]]) -> dict[str, float]:
    usable = [
        point
        for point in points
        if point.get("generated_tokens", 0.0) > 0.0 and point.get("success_rate", 0.0) > 0.0
    ]
    by_x: dict[float, float] = {}
    for point in usable:
        x = float(point["generated_tokens"])
        y = float(point["peak_extra_allocated_mb"])
        by_x[x] = max(y, by_x.get(x, 0.0))

    xs = sorted(by_x)
    if len(xs) < 2:
        return {
            "memory_growth_mb_per_token": 0.0,
            "memory_growth_mb_per_100_tokens": 0.0,
            "fit_points": float(len(xs)),
        }

    ys = [by_x[x] for x in xs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        slope = 0.0
    else:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    slope = max(0.0, slope)
    return {
        "memory_growth_mb_per_token": slope,
        "memory_growth_mb_per_100_tokens": slope * 100.0,
        "fit_points": float(len(xs)),
    }


def summarize_cache_stress(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_group(rows)
    by_max_new_tokens: dict[str, Any] = {}
    points: list[dict[str, float]] = []

    for max_new_tokens in sorted({int(row["max_new_tokens"]) for row in rows}):
        token_rows = [row for row in rows if int(row["max_new_tokens"]) == max_new_tokens]
        token_summary = summarize_group(token_rows)
        by_max_new_tokens[str(max_new_tokens)] = token_summary
        points.append(
            {
                "max_new_tokens": float(max_new_tokens),
                "generated_tokens": float(token_summary.get("avg_generated_ratio", 0.0)) * max_new_tokens,
                "peak_extra_allocated_mb": float(token_summary.get("peak_extra_allocated_mb", 0.0)),
                "success_rate": float(token_summary.get("success_rate", 0.0)),
            }
        )

    summary.update(fit_memory_growth(points))
    summary["by_max_new_tokens"] = by_max_new_tokens
    summary["growth_points"] = points
    return summary


def summarize_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(row.get(field, "")) for row in rows if str(row.get(field, "")).strip()})
    return {
        value: summarize_group([row for row in rows if str(row.get(field, "")) == value])
        for value in values
    }


def summarize_results(rows: list[dict[str, Any]], engine_kind: str, init_memory: dict[str, float]) -> dict[str, Any]:
    suites: dict[str, Any] = {}
    for suite in ALL_SUITES:
        suite_rows = [row for row in rows if row["suite"] == suite]
        if not suite_rows:
            continue
        by_batch_size: dict[str, Any] = {}
        for batch_size in sorted({int(row["batch_size"]) for row in suite_rows}):
            group_rows = [row for row in suite_rows if int(row["batch_size"]) == batch_size]
            if suite == SUITE_CACHE_STRESS:
                by_batch_size[str(batch_size)] = summarize_cache_stress(group_rows)
            else:
                by_batch_size[str(batch_size)] = summarize_group(group_rows)
        summary = summarize_cache_stress(suite_rows) if suite == SUITE_CACHE_STRESS else summarize_group(suite_rows)
        if suite in {SUITE_DECODE, SUITE_MIXED, SUITE_SERVING}:
            best_key = max(
                by_batch_size,
                key=lambda key: float(by_batch_size[key].get("tokens_per_s", 0.0)),
            )
            summary["best_batch_size"] = int(best_key)
            summary["best"] = by_batch_size[best_key]
        if suite == SUITE_TTFT:
            best_key = min(
                by_batch_size,
                key=lambda key: float(by_batch_size[key].get("avg_latency_s", float("inf"))),
            )
            summary["best_batch_size"] = int(best_key)
            summary["best"] = by_batch_size[best_key]
        if suite == SUITE_CACHE_STRESS:
            viable = [
                key
                for key, item in by_batch_size.items()
                if float(item.get("success_rate", 0.0)) >= 0.80 and float(item.get("fit_points", 0.0)) >= 2
            ]
            if viable:
                primary_key = max(viable, key=lambda key: int(key))
            else:
                primary_key = max(by_batch_size, key=lambda key: int(key))
            summary["primary_batch_size"] = int(primary_key)
            summary["primary"] = by_batch_size[primary_key]
        suites[suite] = {
            **summary,
            "by_batch_size": by_batch_size,
            "by_prompt_length_bucket": summarize_by_field(suite_rows, "prompt_length_bucket"),
            "by_workload_type": summarize_by_field(suite_rows, "workload_type"),
        }

    overall = summarize_group(rows)
    overall.update(
        {
            "engine": engine_kind,
            "init_allocated_mb": init_memory["allocated_mb"],
            "init_reserved_mb": init_memory["reserved_mb"],
        }
    )
    return {
        "engine": engine_kind,
        "overall": overall,
        "suites": suites,
    }


def run_global_warmup(engine, args: argparse.Namespace) -> None:
    warmup_iters = max(int(args.warmup_iters), 0)
    if warmup_iters <= 0:
        return
    prompt = (
        "Warmup request WARMUP-KERNEL-INIT. "
        "Write one short sentence about GPU inference warmup."
    )
    for _ in range(warmup_iters):
        try:
            engine.generate(
                prompts=[prompt],
                max_new_tokens=4,
                batch_size=1,
                suite_name=None,
            )
        except Exception:
            traceback.print_exc()
            break
    synchronize(args.device)
    reset_peak(args.device)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_value(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {key: parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def isolated_suite_worker_cmd(args: argparse.Namespace, suite: str, output_dir: Path) -> list[str]:
    cmd = worker_cmd(args, output_dir)
    suite_index = cmd.index("--suites") + 1
    cmd[suite_index] = suite
    isolation_index = cmd.index("--suite-isolation") + 1
    cmd[isolation_index] = "shared"
    return cmd


def run_isolated_suite_workers(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    set_cache_env()
    set_random_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = selected_suites(args.suites)
    print("=" * 72)
    print("ENGINE: student")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}  dtype={args.dtype}  attn={args.attn_implementation}  seed={args.seed}")
    print(f"Suite isolation: process ({len(suites)} fresh worker processes)")
    print("=" * 72)

    all_rows: list[dict[str, Any]] = []
    suite_summaries: list[dict[str, Any]] = []
    suite_runtime_env: dict[str, Any] = collect_runtime_env(args)
    total_init_time_s = 0.0
    worker_env = os.environ.copy()
    worker_env["PYTHONHASHSEED"] = str(int(args.seed))

    for suite in suites:
        suite_dir = output_dir / f"suite_{suite}"
        cmd = isolated_suite_worker_cmd(args, suite, suite_dir)
        print("\n" + "#" * 72, flush=True)
        print(f"Running isolated suite worker: {suite}", flush=True)
        print("#" * 72, flush=True)
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                env=worker_env,
                timeout=args.worker_timeout_s if args.worker_timeout_s > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f"suite worker {suite} timed out after {exc.timeout:.0f}s") from exc
        if completed.returncode != 0:
            raise SystemExit(f"suite worker {suite} failed with exit code {completed.returncode}")

        suite_summary = load_summary(suite_dir / "summary.json")
        suite_summaries.append(suite_summary)
        all_rows.extend(load_csv_rows(suite_dir / "results.csv"))
        total_init_time_s += float(suite_summary.get("init_time_s", 0.0))
        if not suite_runtime_env or suite_runtime_env.get("gpu_name") in {"N/A", ""}:
            suite_runtime_env = suite_summary.get("runtime_env", suite_runtime_env)

    init_memory = {
        "allocated_mb": max(
            (float(item.get("overall", {}).get("init_allocated_mb", 0.0)) for item in suite_summaries),
            default=0.0,
        ),
        "reserved_mb": max(
            (float(item.get("overall", {}).get("init_reserved_mb", 0.0)) for item in suite_summaries),
            default=0.0,
        ),
    }
    summary = summarize_results(all_rows, "student", init_memory)
    summary["init_time_s"] = total_init_time_s
    summary["model"] = args.model
    summary["device"] = args.device
    summary["dtype"] = args.dtype
    summary["attn_implementation"] = args.attn_implementation
    summary["runtime_env"] = suite_runtime_env
    summary["run_config"] = collect_run_config(args)
    summary["data_fingerprints"] = collect_data_fingerprints(args)
    summary["suite_isolation"] = "process"
    summary["suite_init_times_s"] = {
        suite: float(item.get("init_time_s", 0.0))
        for suite, item in zip(suites, suite_summaries)
    }

    save_csv(all_rows, output_dir / "results.csv")
    write_json(output_dir / "summary.json", summary)
    print_worker_summary(summary)
    return summary


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.suite_isolation == "process" and len(selected_suites(args.suites)) > 1:
        return run_isolated_suite_workers(args, Path(args.output_dir))

    set_cache_env()
    set_random_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_cuda(args.device)
    print("=" * 72)
    print("ENGINE: student")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}  dtype={args.dtype}  attn={args.attn_implementation}  seed={args.seed}")
    print("=" * 72)

    init_start = time.perf_counter()
    tokenizer = load_tokenizer(args)
    engine = load_engine(args)
    synchronize(args.device)
    init_time_s = time.perf_counter() - init_start
    init_memory = memory_snapshot(args.device)
    reset_peak(args.device)

    print(
        f"Initialized in {init_time_s:.2f}s, "
        f"allocated={init_memory['allocated_mb']:.0f} MB, "
        f"reserved={init_memory['reserved_mb']:.0f} MB"
    )
    if int(args.warmup_iters) > 0:
        print(f"Warmup: {int(args.warmup_iters)} global generate call(s)")
        run_global_warmup(engine, args)

    all_rows: list[dict[str, Any]] = []
    for suite in selected_suites(args.suites):
        all_rows.extend(run_suite(engine, tokenizer, args, suite, init_memory))

    summary = summarize_results(all_rows, "student", init_memory)
    summary["init_time_s"] = init_time_s
    summary["model"] = args.model
    summary["device"] = args.device
    summary["dtype"] = args.dtype
    summary["attn_implementation"] = args.attn_implementation
    summary["runtime_env"] = collect_runtime_env(args)
    summary["run_config"] = collect_run_config(args)
    summary["data_fingerprints"] = collect_data_fingerprints(args)

    save_csv(all_rows, output_dir / "results.csv")
    write_json(output_dir / "summary.json", summary)
    print_worker_summary(summary)
    return summary


def print_worker_summary(summary: dict[str, Any]) -> None:
    print("\nWORKER SUMMARY")
    print("-" * 72)
    overall = summary["overall"]
    env = summary.get("runtime_env", {})
    print(
        f"engine={summary['engine']}  gpu={env.get('gpu_name', 'N/A')}  "
        f"runtime={overall.get('runtime_success_rate', overall.get('success_rate', 0.0)):.3f}  "
        f"valid={overall.get('success_rate', 0.0):.3f}  "
        f"peak={overall.get('peak_allocated_mb', 0.0):.0f} MB"
    )
    for suite, item in summary.get("suites", {}).items():
        extra = ""
        if "best_batch_size" in item:
            extra = f" best_bs={item.get('best_batch_size')}"
            best = item.get("best", {})
            if isinstance(best, dict) and "tokens_per_s" in best:
                extra += f" best_tps={float(best.get('tokens_per_s', 0.0)):.1f}"
        if "primary_batch_size" in item:
            extra = f" primary_bs={item.get('primary_batch_size')}"
        metric = f"tps={item.get('tokens_per_s', 0.0):.1f}"
        if suite == SUITE_TTFT:
            metric = f"lat={item.get('avg_latency_s', 0.0):.3f}s"
        print(
            f"{suite:<21} valid={item.get('accuracy', 0.0):.3f}  "
            f"{metric}{extra}"
        )


def worker_cmd(args: argparse.Namespace, output_dir: Path) -> list[str]:
    script = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--engine",
        "student",
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--suites",
        args.suites,
        "--output-dir",
        str(output_dir),
        "--long-context-data",
        args.long_context_data,
        "--decode-throughput-data",
        args.decode_throughput_data,
        "--ttft-data",
        args.ttft_data,
        "--serving-schedule-data",
        args.serving_schedule_data,
        "--mixed-serving-data",
        args.mixed_serving_data,
        "--decode-cache-stress-data",
        args.decode_cache_stress_data,
        "--max-new-tokens-long",
        str(args.max_new_tokens_long),
        "--max-new-tokens-decode",
        str(args.max_new_tokens_decode),
        "--max-new-tokens-ttft",
        str(args.max_new_tokens_ttft),
        "--max-new-tokens-serving",
        str(args.max_new_tokens_serving),
        "--max-new-tokens-mixed",
        str(args.max_new_tokens_mixed),
        "--max-new-tokens-cache-stress",
        args.max_new_tokens_cache_stress,
        "--decode-batch-sizes",
        args.decode_batch_sizes,
        "--ttft-batch-sizes",
        args.ttft_batch_sizes,
        "--serving-fallback-batch-size",
        str(args.serving_fallback_batch_size),
        "--mixed-batch-sizes",
        args.mixed_batch_sizes,
        "--cache-stress-batch-sizes",
        args.cache_stress_batch_sizes,
        "--seed",
        str(args.seed),
        "--warmup-iters",
        str(args.warmup_iters),
        "--batch-warmup-iters",
        str(args.batch_warmup_iters),
        "--timed-repeats",
        str(args.timed_repeats),
        "--suite-isolation",
        args.suite_isolation,
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def load_summary(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_summary(path: Path) -> dict[str, Any]:
    payload = load_summary(path)
    if "baseline" in payload and isinstance(payload["baseline"], dict):
        return payload["baseline"]
    return payload


def validate_baseline_fingerprints(args: argparse.Namespace, baseline_summary: dict[str, Any]) -> None:
    expected = collect_data_fingerprints(args)
    baseline_fingerprints = baseline_summary.get("data_fingerprints")
    if not isinstance(baseline_fingerprints, dict) or not baseline_fingerprints:
        message = (
            "Baseline summary has no data fingerprints. The benchmark data may have changed; "
            "please regenerate the baseline summary on the target server."
        )
        if args.allow_stale_baseline:
            print(f"WARNING: {message}", flush=True)
            return
        raise SystemExit(message + " Use --allow-stale-baseline only for debugging.")

    mismatches: list[str] = []
    for suite, expected_item in expected.items():
        baseline_item = baseline_fingerprints.get(suite)
        if not isinstance(baseline_item, dict):
            mismatches.append(f"{suite}: missing in baseline")
            continue
        if baseline_item.get("sha256") != expected_item.get("sha256"):
            mismatches.append(
                f"{suite}: current {expected_item.get('name')} sha256={expected_item.get('sha256', '')[:12]} "
                f"!= baseline sha256={str(baseline_item.get('sha256', ''))[:12]}"
            )

    if mismatches:
        message = (
            "Baseline summary does not match the current benchmark data:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
            + "\nRegenerate the baseline summary on the same target server before scoring."
        )
        if args.allow_stale_baseline:
            print(f"WARNING: {message}", flush=True)
            return
        raise SystemExit(message + "\nUse --allow-stale-baseline only for debugging.")


def validate_baseline_run_config(args: argparse.Namespace, baseline_summary: dict[str, Any]) -> None:
    baseline_config = baseline_summary.get("run_config")
    if not isinstance(baseline_config, dict) or not baseline_config:
        message = (
            "Baseline summary has no run_config. Please regenerate the baseline summary "
            "with the current benchmark runner."
        )
        if args.allow_stale_baseline:
            print(f"WARNING: {message}", flush=True)
            return
        raise SystemExit(message + " Use --allow-stale-baseline only for debugging.")

    current_config = collect_run_config(args)
    selected = selected_suites(args.suites)
    baseline_suites = selected_suites(str(baseline_config.get("suites", "")))
    missing_suites = [suite for suite in selected if suite not in baseline_suites]
    mismatches = [f"suites: missing in baseline: {', '.join(missing_suites)}"] if missing_suites else []

    keys = {"seed", "warmup_iters", "batch_warmup_iters", "timed_repeats", "suite_isolation"}
    per_suite_keys = {
        SUITE_LONG: {"max_new_tokens_long"},
        SUITE_DECODE: {"decode_batch_sizes", "max_new_tokens_decode"},
        SUITE_TTFT: {"ttft_batch_sizes", "max_new_tokens_ttft"},
        SUITE_SERVING: {"serving_fallback_batch_size", "max_new_tokens_serving"},
        SUITE_MIXED: {"mixed_batch_sizes", "max_new_tokens_mixed"},
        SUITE_CACHE_STRESS: {"cache_stress_batch_sizes", "max_new_tokens_cache_stress"},
    }
    for suite in selected:
        keys.update(per_suite_keys.get(suite, set()))

    for key in sorted(keys):
        current_value = current_config.get(key)
        baseline_value = baseline_config.get(key)
        if str(current_value) != str(baseline_value):
            mismatches.append(f"{key}: current={current_value!r}, baseline={baseline_value!r}")

    if mismatches:
        message = (
            "Baseline summary run_config does not match this benchmark command:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
            + "\nUse the same seed, token budgets, batch sizes, warmup, and timed repeats as baseline."
        )
        if args.allow_stale_baseline:
            print(f"WARNING: {message}", flush=True)
            return
        raise SystemExit(message + "\nUse --allow-stale-baseline only for debugging.")


def run_static_validation(args: argparse.Namespace) -> None:
    if bool(args.skip_validation):
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validate_engine.py"),
        "--skip-load",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_controller(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_static_validation(args)

    baseline_path = Path(resolve_project_path(args.baseline_summary))
    if not baseline_path.exists():
        raise SystemExit(f"Baseline summary not found: {baseline_path}")
    baseline_summary = load_baseline_summary(baseline_path)
    validate_baseline_fingerprints(args, baseline_summary)
    validate_baseline_run_config(args, baseline_summary)

    worker_dir = output_dir / "student"
    cmd = worker_cmd(args, worker_dir)
    worker_env = os.environ.copy()
    worker_env["PYTHONHASHSEED"] = str(int(args.seed))
    print("\n" + "#" * 72, flush=True)
    print("Running student in an isolated process", flush=True)
    print("#" * 72, flush=True)
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            env=worker_env,
            timeout=args.worker_timeout_s if args.worker_timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"student worker timed out after {exc.timeout:.0f}s") from exc
    if completed.returncode != 0:
        raise SystemExit(f"student worker failed with exit code {completed.returncode}")

    student_summary = load_summary(worker_dir / "summary.json")

    score = compute_final_score(student=student_summary, baseline=baseline_summary)
    final_payload = {
        "baseline": baseline_summary,
        "student": student_summary,
        "score": score,
        "baseline_summary_path": str(baseline_path),
    }
    write_json(output_dir / "final_summary.json", final_payload)
    text = format_final_summary(baseline_summary, student_summary, score, baseline_path)
    write_text(output_dir / "final_summary.txt", text)
    print(text)


def format_final_summary(
    baseline: dict[str, Any],
    student: dict[str, Any],
    score: dict[str, float],
    baseline_path: Path | str | None = None,
) -> str:
    b_suites = baseline.get("suites", {})
    s_suites = student.get("suites", {})
    b_decode = suite_score_view(b_suites.get(SUITE_DECODE, {}))
    s_decode = suite_score_view(s_suites.get(SUITE_DECODE, {}))
    b_ttft = suite_score_view(b_suites.get(SUITE_TTFT, {}))
    s_ttft = suite_score_view(s_suites.get(SUITE_TTFT, {}))
    if not b_ttft or not s_ttft:
        b_ttft = suite_score_view(b_suites.get(SUITE_MIXED, b_suites.get(SUITE_LONG, {})))
        s_ttft = suite_score_view(s_suites.get(SUITE_MIXED, s_suites.get(SUITE_LONG, {})))
    b_mixed = suite_score_view(b_suites.get(SUITE_MIXED, {}))
    s_mixed = suite_score_view(s_suites.get(SUITE_MIXED, {}))
    b_serving = suite_score_view(b_suites.get(SUITE_SERVING, b_suites.get(SUITE_MIXED, {})))
    s_serving = suite_score_view(s_suites.get(SUITE_SERVING, s_suites.get(SUITE_MIXED, {})))
    b_cache = suite_score_view(b_suites.get(SUITE_CACHE_STRESS, {}))
    s_cache = suite_score_view(s_suites.get(SUITE_CACHE_STRESS, {}))
    b_mem = baseline.get("overall", {})
    s_mem = student.get("overall", {})
    env = student.get("runtime_env", {})
    run_config = student.get("run_config", {})
    decode_speedup = safe_div(float(s_decode.get("tokens_per_s", 0.0)), float(b_decode.get("tokens_per_s", 0.0)))
    ttft_speedup = safe_div(float(b_ttft.get("avg_latency_s", 0.0)), float(s_ttft.get("avg_latency_s", 0.0)))
    ttft_p95_speedup = safe_div(
        float(b_ttft.get("p95_latency_s", b_ttft.get("avg_latency_s", 0.0))),
        float(s_ttft.get("p95_latency_s", s_ttft.get("avg_latency_s", 0.0))),
    )
    serving_speedup = safe_div(float(s_serving.get("tokens_per_s", 0.0)), float(b_serving.get("tokens_per_s", 0.0)))
    mixed_speedup = safe_div(float(s_mixed.get("tokens_per_s", 0.0)), float(b_mixed.get("tokens_per_s", 0.0)))
    cache_speedup = safe_div(float(s_cache.get("tokens_per_s", 0.0)), float(b_cache.get("tokens_per_s", 0.0)))
    method_decode_scaling = batch_scaling_factor(s_suites.get(SUITE_DECODE, {}))
    baseline_cache_growth = float(b_cache.get("memory_growth_mb_per_100_tokens", 0.0))
    method_cache_growth = float(s_cache.get("memory_growth_mb_per_100_tokens", 0.0))
    baseline_cache_peak = float(b_cache.get("peak_extra_allocated_mb", 0.0))
    method_cache_peak = float(s_cache.get("peak_extra_allocated_mb", 0.0))
    cache_growth_usable = bool(score.get("cache_growth_metric_usable", 0.0))
    cache_peak_saving = safe_div(
        baseline_cache_peak - method_cache_peak,
        baseline_cache_peak,
    )
    cache_memory_metric = str(score.get("cache_memory_metric", "growth_slope" if cache_growth_usable else "peak_extra_fallback"))
    guard_notes = [
        str(score.get(key, ""))
        for key in [
            "long_context_realism_reason",
            "decode_realism_reason",
            "ttft_realism_reason",
            "serving_realism_reason",
            "mixed_realism_reason",
            "cache_realism_reason",
        ]
        if str(score.get(key, "")).strip()
    ]
    guard_status = "OK" if not guard_notes else "; ".join(guard_notes)
    baseline_text = str(baseline_path) if baseline_path is not None else ""
    suites_text = str(run_config.get("suites", ""))
    limit_text = "None" if run_config.get("limit") is None else str(run_config.get("limit"))
    local_only_text = str(run_config.get("local_files_only", ""))
    gpu_text = str(env.get("gpu_name", "N/A"))
    host_text = str(env.get("hostname", ""))
    seed_text = str(run_config.get("seed", env.get("seed", "")))
    cuda_visible = str(env.get("cuda_visible_devices", ""))
    if not cuda_visible:
        cuda_visible = "unset"

    lines = [
        "",
        "=" * 72,
        "INFERENCE OPTIMIZATION BENCHMARK RESULT",
        "=" * 72,
        f"Score profile: {score.get('scoring_profile', 'vllm_reference_teacher_v7_component_tiered')} "
        f"(full score speedup bar={float(score.get('full_score_speedup_bar', 1.0)):.2f}x)",
        f"Model: {student.get('model', run_config.get('model', ''))}",
        f"Runtime: host={host_text or 'N/A'}  gpu={gpu_text}  CUDA_VISIBLE_DEVICES={cuda_visible}",
        f"Config: device={student.get('device', run_config.get('device', ''))}  "
        f"dtype={student.get('dtype', run_config.get('dtype', ''))}  "
        f"attn={student.get('attn_implementation', run_config.get('attn_implementation', ''))}  "
        f"seed={seed_text}  "
        f"local_files_only={local_only_text}",
        f"Workload: suites={suites_text}  limit={limit_text}  baseline={baseline_text}",
        "-" * 72,
        f"Long Context Correctness  {score['correctness_score']:6.2f} / 30   "
        f"partial={score.get('long_context_partial_score', score['long_context_accuracy']):.3f}  "
        f"exact={score['long_context_accuracy']:.3f}",
        f"Decode TPS                {score.get('decode_score', 0.0):6.2f} / 25   "
        f"tps={float(s_decode.get('tokens_per_s', 0.0)):.1f}  speedup={decode_speedup:.2f}x  "
        f"valid={float(s_decode.get('valid_output_rate', 0.0)):.3f}",
        f"TTFT / Prefill Latency    {score.get('ttft_score', 0.0):6.2f} / 20   "
        f"avg={float(s_ttft.get('avg_latency_s', 0.0)):.3f}s({ttft_speedup:.2f}x)  "
        f"p95={float(s_ttft.get('p95_latency_s', 0.0)):.3f}s({ttft_p95_speedup:.2f}x)  "
        f"buckets={int(float(score.get('ttft_bucket_latency_bucket_count', 0.0)))}",
        f"TTFT Breakdown             bucket_avg={score.get('ttft_bucket_latency_score', 0.0):5.2f}/12  "
        f"bucket_p95={score.get('ttft_p95_latency_score', 0.0):5.2f}/6  "
        f"quality={score.get('ttft_quality_score', 0.0):4.2f}/2  "
        f"valid={float(s_ttft.get('valid_output_rate', 0.0)):.3f}",
        f"Serving / Scheduling      {score.get('serving_score', 0.0):6.2f} / 15   "
        f"tps={float(s_serving.get('tokens_per_s', 0.0)):.1f}  speedup={serving_speedup:.2f}x  "
        f"p95={float(s_serving.get('p95_latency_s', 0.0)):.3f}s  "
        f"iface={str(s_serving.get('serving_interface', 'generate') or 'generate')}",
        f"Runtime Robustness        {score['stability_score']:6.2f} / 10   "
        f"runtime={float(score.get('runtime_success_rate', score['success_rate'])):.3f}  "
        f"valid={score['success_rate']:.3f}",
        f"Diagnostics                not scored   "
        f"batch_scale={method_decode_scaling:.2f}x  "
        f"mixed_tps={float(s_mixed.get('tokens_per_s', 0.0)):.1f}({mixed_speedup:.2f}x)  "
        f"prefix_shared={float(score.get('prefix_shared_vs_regular_tokens_ratio', 0.0)):.2f}x  "
        f"copy_prefix={float(s_mem.get('avg_copied_prompt_prefix_ratio', 0.0)):.3f}",
        f"Cache/Memory Diagnostic    not scored   "
        f"cache_tps={float(s_cache.get('tokens_per_s', 0.0)):.1f}({cache_speedup:.2f}x)  "
        f"growth={method_cache_growth:.2f} MB/100tok  "
        f"extra={float(s_mem.get('peak_extra_allocated_mb', 0.0)):.0f} MB  "
        f"cache_metric={cache_memory_metric}  cache_peak_saving={cache_peak_saving * 100:.1f}%",
        f"Realism Guard: {guard_status}",
        f"Component Tiering         raw={float(score.get('raw_component_score', score['final_score'])):.2f}  "
        f"before_cap={float(score.get('component_tiered_score_before_cap', score['final_score'])):.2f}  "
        f"full_bar={float(score.get('full_score_speedup_bar', 1.0)):.2f}x",
        "-" * 72,
        f"FINAL SCORE: {score['final_score']:.2f} / 100   CAP: {score['cap']:.0f}",
        f"SERVER/GPU: host={host_text or 'N/A'} | gpu={gpu_text} | CUDA_VISIBLE_DEVICES={cuda_visible}",
        f"RUN CONFIG: model={student.get('model', run_config.get('model', ''))} | "
        f"dtype={student.get('dtype', run_config.get('dtype', ''))} | "
        f"attn={student.get('attn_implementation', run_config.get('attn_implementation', ''))} | "
        f"seed={seed_text} | "
        f"suites={suites_text} | limit={limit_text}",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    if args.worker:
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()

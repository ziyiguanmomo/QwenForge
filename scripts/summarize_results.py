#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import compute_final_score, safe_div, suite_score_view


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="?", default="results/final_eval/final_summary.json")
    return parser.parse_args()


def suite_view(suites: dict[str, Any], name: str) -> dict[str, Any]:
    return suite_score_view(suites.get(name, {}))


def main() -> None:
    args = parse_args()
    path = Path(args.summary)
    payload = json.loads(path.read_text(encoding="utf-8"))
    student = payload["student"]
    baseline = payload["baseline"]
    stored_score = payload.get("score", {})
    score = compute_final_score(student, baseline)

    student_suites = student.get("suites", {})
    baseline_suites = baseline.get("suites", {})
    s_decode = suite_view(student_suites, "decode_throughput")
    b_decode = suite_view(baseline_suites, "decode_throughput")
    s_ttft = suite_view(student_suites, "ttft_prefill")
    b_ttft = suite_view(baseline_suites, "ttft_prefill")
    if not s_ttft or not b_ttft:
        s_ttft = suite_view(student_suites, "mixed_serving") or suite_view(student_suites, "long_context")
        b_ttft = suite_view(baseline_suites, "mixed_serving") or suite_view(baseline_suites, "long_context")
    s_serving = suite_view(student_suites, "serving_schedule") or suite_view(student_suites, "mixed_serving")
    b_serving = suite_view(baseline_suites, "serving_schedule") or suite_view(baseline_suites, "mixed_serving")

    env = student.get("runtime_env", {})
    run_config = student.get("run_config", {})
    gpu = env.get("gpu_name", "N/A")
    host = env.get("hostname", "N/A")
    cuda_visible = env.get("cuda_visible_devices") or "unset"
    suites = run_config.get("suites", "")
    limit = "None" if run_config.get("limit") is None else str(run_config.get("limit"))
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

    print(f"Summary: {path}")
    print("=" * 72)
    print(
        f"Scoring profile:       {score.get('scoring_profile', 'vllm_reference_teacher_v7_component_tiered')} "
        f"(full speedup bar={float(score.get('full_score_speedup_bar', 1.0)):.2f}x)"
    )
    print(f"FINAL SCORE: {score['final_score']:.2f} / 100")
    if stored_score:
        stored_final = float(stored_score.get("final_score", score["final_score"]))
        if abs(stored_final - score["final_score"]) > 1e-6:
            print(f"Stored score:          {stored_final:.2f} / 100")
            print("Note: displayed score was recomputed with the current metrics.py.")

    print("-" * 72)
    print(
        f"Long Context:          {score['correctness_score']:.2f} / 30   "
        f"partial={score.get('long_context_partial_score', score['long_context_accuracy']):.3f}  "
        f"exact={score['long_context_accuracy']:.3f}"
    )
    print(
        f"Decode TPS:            {score.get('decode_score', 0.0):.2f} / 25   "
        f"tps={float(s_decode.get('tokens_per_s', 0.0)):.1f}  "
        f"speedup={safe_div(float(s_decode.get('tokens_per_s', 0.0)), float(b_decode.get('tokens_per_s', 0.0))):.2f}x"
    )
    print(
        f"TTFT / Prefill:        {score.get('ttft_score', 0.0):.2f} / 20   "
        f"lat={float(s_ttft.get('avg_latency_s', 0.0)):.3f}s  "
        f"speedup={safe_div(float(b_ttft.get('avg_latency_s', 0.0)), float(s_ttft.get('avg_latency_s', 0.0))):.2f}x"
    )
    print(
        f"Serving / Scheduling:  {score.get('serving_score', 0.0):.2f} / 15   "
        f"tps={float(s_serving.get('tokens_per_s', 0.0)):.1f}  "
        f"speedup={safe_div(float(s_serving.get('tokens_per_s', 0.0)), float(b_serving.get('tokens_per_s', 0.0))):.2f}x  "
        f"iface={str(s_serving.get('serving_interface', 'generate') or 'generate')}"
    )
    print(f"Runtime Robustness:    {score['stability_score']:.2f} / 10   runtime={score.get('runtime_success_rate', 0.0):.3f}")
    print(
        "Diagnostics:           "
        f"batch_norm={score.get('batch_scaling_norm', 0.0):.3f}, "
        f"mixed_norm={score.get('mixed_speed_norm', 0.0):.3f}, "
        f"prefix_norm={score.get('prefix_reuse_norm', 0.0):.3f}, "
        f"cache_mem_norm={score.get('cache_memory_norm', 0.0):.3f}"
    )
    print(
        "Memory Diagnostic:     "
        f"peak={student.get('overall', {}).get('peak_allocated_mb', 0.0):.0f} MB, "
        f"extra={student.get('overall', {}).get('peak_extra_allocated_mb', 0.0):.0f} MB"
    )
    print(f"Realism Guard:         {'OK' if not guard_notes else '; '.join(guard_notes)}")
    print(
        "Component Tiering:     "
        f"raw={float(score.get('raw_component_score', score['final_score'])):.2f}, "
        f"before_cap={float(score.get('component_tiered_score_before_cap', score['final_score'])):.2f}, "
        f"full_bar={float(score.get('full_score_speedup_bar', 1.0)):.2f}x"
    )
    print("-" * 72)
    print(f"FINAL SCORE: {score['final_score']:.2f} / 100   CAP: {score['cap']:.0f}")
    print(f"SERVER/GPU: host={host} | gpu={gpu} | CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(
        "RUN CONFIG: "
        f"model={student.get('model', run_config.get('model', ''))} | "
        f"dtype={student.get('dtype', run_config.get('dtype', ''))} | "
        f"attn={student.get('attn_implementation', run_config.get('attn_implementation', ''))} | "
        f"suites={suites} | limit={limit}"
    )


if __name__ == "__main__":
    main()

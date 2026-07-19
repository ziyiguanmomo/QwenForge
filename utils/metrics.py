from __future__ import annotations

from typing import Any


CACHE_GROWTH_USABLE_THRESHOLD = 1e-9
INIT_MEMORY_PENALTY_RATIO = 1.30
INIT_MEMORY_PENALTY = 0.80
TOTAL_MEMORY_GUARD_RATIO = 1.30
TOTAL_MEMORY_GUARD_PENALTY = 0.80
TEACHING_MEMORY_BUDGET_MB = 7600.0
REFERENCE_FULL_SCORE_RATIO = 1.60
BLACKBOX_FULL_SCORE_SPEEDUP = REFERENCE_FULL_SCORE_RATIO
HIGH_SCORE_STRETCH_POINTS = [
    (0.0, 0.0),
    (100.0, 100.0),
]
REALISM_HARD_SPEEDUP = 8.0
REALISM_SOFT_SPEEDUP = 5.0
REALISM_HARD_TOKENS_PER_S = 5000.0
REALISM_SOFT_TOKENS_PER_S = 3000.0
LONG_CONTEXT_HARD_LATENCY_SPEEDUP = 20.0
LONG_CONTEXT_SOFT_LATENCY_SPEEDUP = 10.0
TTFT_HARD_LATENCY_SPEEDUP = 100.0
TTFT_SOFT_LATENCY_SPEEDUP = 50.0
TTFT_HARD_ABSOLUTE_LATENCY_S = 0.0015
TTFT_SOFT_ABSOLUTE_LATENCY_S = 0.005
TTFT_LENGTH_BUCKET_ORDER = ("short", "medium", "long", "extra_long")


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(float(den)) < 1e-12:
        return default
    return float(num) / float(den)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def output_token_count(tokenizer, text: str) -> int:
    ids = tokenizer(text or "", add_special_tokens=False).get("input_ids", [])
    return int(len(ids))


def substring_score(generated_text: str, references: list[str]) -> float:
    if not references:
        return 1.0
    pred = (generated_text or "").lower()
    return mean([1.0 if str(ref).lower() in pred else 0.0 for ref in references])


def required_substring_score(generated_text: str, required: list[str]) -> float:
    if not required:
        return 1.0
    pred = generated_text or ""
    return mean([1.0 if str(item) in pred else 0.0 for item in required])


def keyword_score(generated_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    pred = (generated_text or "").lower()
    return mean([1.0 if str(item).lower() in pred else 0.0 for item in keywords])


def generation_penalty(avg_ratio: float, min_ratio: float = 0.80) -> float:
    return clamp01(safe_div(avg_ratio, min_ratio, default=0.0))


def speed_norm(method_tokens_per_s: float, baseline_tokens_per_s: float, full_score_speedup: float = 1.5) -> float:
    speedup = safe_div(method_tokens_per_s, baseline_tokens_per_s, default=0.0)
    return clamp01(speedup / full_score_speedup)


def piecewise_linear(value: float, points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    x = float(value)
    ordered = sorted(points)
    if x <= ordered[0][0]:
        return clamp01(ordered[0][1])
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x <= x1:
            if abs(x1 - x0) < 1e-12:
                return clamp01(y1)
            t = (x - x0) / (x1 - x0)
            return clamp01(y0 + t * (y1 - y0))
    return clamp01(ordered[-1][1])


def high_score_stretch(raw_score: float) -> float:
    """Keep normal scores intuitive while making 95+ a real stretch zone."""
    points = sorted(HIGH_SCORE_STRETCH_POINTS)
    x = min(max(float(raw_score), points[0][0]), points[-1][0])
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            if abs(x1 - x0) < 1e-12:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


def teaching_speed_norm(method_tokens_per_s: float, baseline_tokens_per_s: float) -> float:
    speedup = safe_div(method_tokens_per_s, baseline_tokens_per_s, default=0.0)
    return piecewise_linear(
        speedup,
        [
            (0.0, 0.0),
            (0.25, 0.20),
            (0.50, 0.55),
            (0.80, 0.85),
            (1.00, 1.00),
        ],
    )


def teaching_latency_norm(method_latency_s: float, baseline_latency_s: float) -> float:
    latency_speedup = safe_div(baseline_latency_s, method_latency_s, default=0.0)
    return piecewise_linear(
        latency_speedup,
        [
            (0.0, 0.0),
            (0.25, 0.20),
            (0.50, 0.55),
            (0.80, 0.85),
            (1.00, 1.00),
        ],
    )


def blackbox_speed_norm(method_tokens_per_s: float, baseline_tokens_per_s: float) -> float:
    speedup = safe_div(method_tokens_per_s, baseline_tokens_per_s, default=0.0)
    return piecewise_linear(
        speedup,
        [
            (0.0, 0.0),
            (0.02, 0.20),
            (0.05, 0.38),
            (0.10, 0.55),
            (0.20, 0.70),
            (0.35, 0.82),
            (0.60, 0.90),
            (0.85, 0.94),
            (1.00, 0.96),
            (1.25, 0.985),
            (REFERENCE_FULL_SCORE_RATIO, 1.00),
        ],
    )


def blackbox_latency_norm(method_latency_s: float, baseline_latency_s: float) -> float:
    latency_speedup = safe_div(baseline_latency_s, method_latency_s, default=0.0)
    return piecewise_linear(
        latency_speedup,
        [
            (0.0, 0.0),
            (0.02, 0.20),
            (0.05, 0.38),
            (0.10, 0.55),
            (0.25, 0.70),
            (0.50, 0.84),
            (0.85, 0.92),
            (1.00, 0.95),
            (1.25, 0.98),
            (REFERENCE_FULL_SCORE_RATIO, 1.00),
        ],
    )


def reference_speed_norm(method_tokens_per_s: float, reference_tokens_per_s: float) -> float:
    return blackbox_speed_norm(method_tokens_per_s, reference_tokens_per_s)


def reference_latency_norm(method_latency_s: float, reference_latency_s: float) -> float:
    return blackbox_latency_norm(method_latency_s, reference_latency_s)


def quality_norm(valid_rate: float, generated_ratio: float, success_rate: float = 1.0) -> float:
    quality = min(
        clamp01(valid_rate),
        generation_penalty(generated_ratio),
        clamp01(success_rate),
    )
    return piecewise_linear(
        quality,
        [
            (0.0, 0.0),
            (0.50, 0.55),
            (0.80, 0.86),
            (0.95, 1.00),
        ],
    )


def suite_quality_norm(suite: dict[str, Any]) -> float:
    return quality_norm(
        float(suite.get("valid_output_rate", 0.0)),
        float(suite.get("avg_generated_ratio", 0.0)),
        float(suite.get("success_rate", 0.0)),
    )


def throughput_realism_guard(method_suite: dict[str, Any], baseline_suite: dict[str, Any], prefix: str) -> dict[str, Any]:
    method_tokens = float(method_suite.get("tokens_per_s", 0.0))
    baseline_tokens = float(baseline_suite.get("tokens_per_s", 0.0))
    speedup = safe_div(method_tokens, baseline_tokens, default=0.0)
    hard_limit = max(REALISM_HARD_TOKENS_PER_S, baseline_tokens * REALISM_HARD_SPEEDUP)
    soft_limit = max(REALISM_SOFT_TOKENS_PER_S, baseline_tokens * REALISM_SOFT_SPEEDUP)

    penalty = 1.0
    reason = ""
    if method_tokens > hard_limit:
        penalty = 0.0
        reason = (
            f"{prefix} throughput is unrealistically high "
            f"({method_tokens:.2f} tok/s > {hard_limit:.2f} tok/s guardrail)"
        )
    elif method_tokens > soft_limit:
        penalty = 0.5
        reason = (
            f"{prefix} throughput is suspiciously high "
            f"({method_tokens:.2f} tok/s > {soft_limit:.2f} tok/s guardrail)"
        )

    return {
        f"{prefix}_realism_penalty": penalty,
        f"{prefix}_realism_reason": reason,
        f"{prefix}_tokens_per_s": method_tokens,
        f"{prefix}_baseline_tokens_per_s": baseline_tokens,
        f"{prefix}_speedup_for_realism": speedup,
        f"{prefix}_realism_hard_limit_tokens_per_s": hard_limit,
        f"{prefix}_realism_soft_limit_tokens_per_s": soft_limit,
    }


def long_context_realism_guard(method_suite: dict[str, Any], baseline_suite: dict[str, Any]) -> dict[str, Any]:
    method_latency = float(method_suite.get("avg_latency_s", 0.0))
    baseline_latency = float(baseline_suite.get("avg_latency_s", 0.0))
    latency_speedup = safe_div(baseline_latency, method_latency, default=0.0)

    penalty = 1.0
    reason = ""
    if method_latency > 0.0 and latency_speedup > LONG_CONTEXT_HARD_LATENCY_SPEEDUP:
        penalty = 0.0
        reason = (
            "long_context latency is unrealistically low for full-prefill inference "
            f"({latency_speedup:.2f}x faster than baseline)"
        )
    elif method_latency > 0.0 and latency_speedup > LONG_CONTEXT_SOFT_LATENCY_SPEEDUP:
        penalty = 0.5
        reason = (
            "long_context latency is suspiciously low for full-prefill inference "
            f"({latency_speedup:.2f}x faster than baseline)"
        )

    return {
        "long_context_realism_penalty": penalty,
        "long_context_realism_reason": reason,
        "long_context_latency_s": method_latency,
        "long_context_baseline_latency_s": baseline_latency,
        "long_context_latency_speedup_for_realism": latency_speedup,
    }


def latency_realism_guard(method_suite: dict[str, Any], baseline_suite: dict[str, Any], prefix: str) -> dict[str, Any]:
    method_latency = float(method_suite.get("avg_latency_s", 0.0))
    baseline_latency = float(baseline_suite.get("avg_latency_s", 0.0))
    latency_speedup = safe_div(baseline_latency, method_latency, default=0.0)
    if prefix == "ttft":
        hard_speedup = TTFT_HARD_LATENCY_SPEEDUP
        soft_speedup = TTFT_SOFT_LATENCY_SPEEDUP
        hard_abs_latency = TTFT_HARD_ABSOLUTE_LATENCY_S
        soft_abs_latency = TTFT_SOFT_ABSOLUTE_LATENCY_S
    else:
        hard_speedup = LONG_CONTEXT_HARD_LATENCY_SPEEDUP
        soft_speedup = LONG_CONTEXT_SOFT_LATENCY_SPEEDUP
        hard_abs_latency = float("inf")
        soft_abs_latency = float("inf")

    penalty = 1.0
    reason = ""
    if method_latency > 0.0 and latency_speedup > hard_speedup and method_latency <= hard_abs_latency:
        penalty = 0.0
        reason = (
            f"{prefix} latency is unrealistically low for real prefill/decode "
            f"({latency_speedup:.2f}x faster than baseline, {method_latency:.4f}s)"
        )
    elif method_latency > 0.0 and latency_speedup > soft_speedup and method_latency <= soft_abs_latency:
        penalty = 0.5
        reason = (
            f"{prefix} latency is suspiciously low for real prefill/decode "
            f"({latency_speedup:.2f}x faster than baseline, {method_latency:.4f}s)"
        )

    return {
        f"{prefix}_realism_penalty": penalty,
        f"{prefix}_realism_reason": reason,
        f"{prefix}_latency_s": method_latency,
        f"{prefix}_baseline_latency_s": baseline_latency,
        f"{prefix}_latency_speedup_for_realism": latency_speedup,
        f"{prefix}_realism_hard_latency_speedup": hard_speedup,
        f"{prefix}_realism_soft_latency_speedup": soft_speedup,
        f"{prefix}_realism_hard_absolute_latency_s": hard_abs_latency,
        f"{prefix}_realism_soft_absolute_latency_s": soft_abs_latency,
    }


def batch_scaling_factor(suite: dict[str, Any]) -> float:
    groups = suite.get("by_batch_size", {})
    if not groups:
        return 0.0
    batch_sizes = sorted(int(key) for key in groups)
    if not batch_sizes:
        return 0.0
    base = groups[str(batch_sizes[0])]
    target = groups[str(batch_sizes[-1])]
    base_tokens = float(base.get("tokens_per_s", 0.0))
    target_tokens = float(target.get("tokens_per_s", 0.0))
    return safe_div(target_tokens, base_tokens, default=0.0)


def batch_scaling_norm(method_suite: dict[str, Any], baseline_suite: dict[str, Any], full_score_ratio: float = 1.5) -> float:
    baseline_scale = batch_scaling_factor(baseline_suite)
    if baseline_scale <= 1e-9:
        return 0.0
    method_scale = batch_scaling_factor(method_suite)
    return clamp01(safe_div(method_scale, baseline_scale, default=0.0) / full_score_ratio)


def teaching_batch_scaling_norm(method_suite: dict[str, Any]) -> float:
    method_scale = batch_scaling_factor(method_suite)
    return piecewise_linear(
        method_scale,
        [
            (1.00, 0.0),
            (1.50, 0.35),
            (2.00, 0.70),
            (2.50, 1.00),
        ],
    )


def memory_norm(method_extra_mb: float, baseline_extra_mb: float, full_score_saving: float = 0.30) -> float:
    if baseline_extra_mb <= 0:
        return 0.0
    saving_ratio = max(0.0, baseline_extra_mb - method_extra_mb) / baseline_extra_mb
    return clamp01(saving_ratio / full_score_saving)


def cache_growth_norm(method_growth: float, baseline_growth: float, full_score_saving: float = 0.50) -> float:
    if baseline_growth <= CACHE_GROWTH_USABLE_THRESHOLD:
        return 0.0
    saving_ratio = max(0.0, baseline_growth - method_growth) / baseline_growth
    return clamp01(saving_ratio / full_score_saving)


def over_budget_norm(peak_total_mb: float, budget_mb: float = TEACHING_MEMORY_BUDGET_MB) -> float:
    if peak_total_mb <= 0.0:
        return 0.0
    if peak_total_mb <= budget_mb * 0.80:
        return 1.0
    if peak_total_mb <= budget_mb:
        return piecewise_linear(peak_total_mb / budget_mb, [(0.80, 1.0), (1.00, 0.70)])
    if peak_total_mb <= budget_mb * 1.10:
        return piecewise_linear(peak_total_mb / budget_mb, [(1.00, 0.70), (1.10, 0.0)])
    return 0.0


def relative_memory_norm(method_peak_total_mb: float, baseline_peak_total_mb: float) -> float:
    ratio = safe_div(method_peak_total_mb, baseline_peak_total_mb, default=0.0)
    if ratio <= 0.0:
        return 0.0
    return piecewise_linear(
        ratio,
        [
            (1.00, 1.00),
            (1.30, 1.00),
            (1.60, 0.60),
            (2.00, 0.0),
        ],
    )


def group_quality_norm(group: dict[str, Any]) -> float:
    if not group:
        return 0.0
    return suite_quality_norm(group)


def grouped_quality_norm(groups: dict[str, Any]) -> float:
    if not groups:
        return 0.0
    return mean([group_quality_norm(group) for group in groups.values()])


def length_bucket_norm(suite: dict[str, Any]) -> float:
    groups = suite.get("by_prompt_length_bucket", {})
    if not groups:
        return suite_quality_norm(suite)
    return grouped_quality_norm(groups)


def ordered_shared_bucket_keys(
    method_groups: dict[str, Any],
    baseline_groups: dict[str, Any],
) -> list[str]:
    shared = {str(key) for key in method_groups} & {str(key) for key in baseline_groups}
    ordered = [key for key in TTFT_LENGTH_BUCKET_ORDER if key in shared]
    ordered.extend(sorted(shared - set(ordered)))
    return ordered


def group_latency_norm(
    method_group: dict[str, Any],
    baseline_group: dict[str, Any],
    latency_key: str,
) -> float:
    method_latency = float(method_group.get(latency_key, method_group.get("avg_latency_s", 0.0)))
    baseline_latency = float(baseline_group.get(latency_key, baseline_group.get("avg_latency_s", 0.0)))
    return reference_latency_norm(method_latency, baseline_latency) * suite_quality_norm(method_group)


def ttft_latency_bucket_stats(
    method_suite: dict[str, Any],
    baseline_suite: dict[str, Any],
    latency_key: str,
    prefix: str,
) -> dict[str, Any]:
    method_groups = method_suite.get("by_prompt_length_bucket", {})
    baseline_groups = baseline_suite.get("by_prompt_length_bucket", {})
    bucket_keys = ordered_shared_bucket_keys(method_groups, baseline_groups)
    bucket_norms: dict[str, float] = {}

    for key in bucket_keys:
        bucket_norms[key] = group_latency_norm(method_groups[key], baseline_groups[key], latency_key)

    if bucket_norms:
        norm = mean(list(bucket_norms.values()))
        count = float(len(bucket_norms))
    else:
        method_view = suite_score_view(method_suite)
        baseline_view = suite_score_view(baseline_suite)
        norm = group_latency_norm(method_view, baseline_view, latency_key)
        count = 0.0

    return {
        f"{prefix}_norm": norm,
        f"{prefix}_bucket_count": count,
        f"{prefix}_by_bucket": bucket_norms,
    }


def cache_length_norm(cache_suite: dict[str, Any]) -> float:
    primary = cache_suite.get("primary", {})
    token_groups = primary.get("by_max_new_tokens", {})
    if token_groups:
        return grouped_quality_norm(token_groups)
    return length_bucket_norm(cache_suite)


def prefix_reuse_stats(student_mixed_suite: dict[str, Any], baseline_mixed_suite: dict[str, Any]) -> dict[str, float]:
    student_groups = student_mixed_suite.get("by_workload_type", {})
    baseline_groups = baseline_mixed_suite.get("by_workload_type", {})
    student_shared = student_groups.get("shared_prefix", {})
    student_regular = student_groups.get("mixed_length", student_mixed_suite.get("best", {}))
    baseline_shared = baseline_groups.get("shared_prefix", {})

    shared_quality = suite_quality_norm(student_shared) if student_shared else 0.0
    shared_speed_norm = teaching_speed_norm(
        float(student_shared.get("tokens_per_s", 0.0)),
        float(baseline_shared.get("tokens_per_s", 0.0)),
    )
    shared_vs_regular = safe_div(
        float(student_shared.get("tokens_per_s", 0.0)),
        float(student_regular.get("tokens_per_s", 0.0)),
        default=0.0,
    )
    balance_norm = piecewise_linear(
        shared_vs_regular,
        [
            (0.30, 0.0),
            (0.70, 0.60),
            (0.90, 1.00),
        ],
    )
    prefix_norm = 0.35 * shared_quality + 0.40 * shared_speed_norm + 0.25 * balance_norm
    return {
        "prefix_shared_quality_norm": shared_quality,
        "prefix_shared_speed_norm": shared_speed_norm,
        "prefix_shared_balance_norm": balance_norm,
        "prefix_shared_vs_regular_tokens_ratio": shared_vs_regular,
        "prefix_reuse_norm": prefix_norm,
    }


def metric_value(primary: dict[str, Any], fallback: dict[str, Any], key: str) -> float:
    if key in primary and primary.get(key) is not None:
        return float(primary.get(key, 0.0))
    return float(fallback.get(key, 0.0))


def init_memory_penalty(
    method_cache: dict[str, Any],
    baseline_cache: dict[str, Any],
    method_overall: dict[str, Any],
    baseline_overall: dict[str, Any],
) -> dict[str, float]:
    method_init = metric_value(method_cache, method_overall, "init_allocated_mb")
    baseline_init = metric_value(baseline_cache, baseline_overall, "init_allocated_mb")
    init_ratio = safe_div(method_init, baseline_init, default=0.0)
    penalty = INIT_MEMORY_PENALTY if baseline_init > 0.0 and init_ratio > INIT_MEMORY_PENALTY_RATIO else 1.0
    return {
        "cache_method_init_allocated_mb": method_init,
        "cache_baseline_init_allocated_mb": baseline_init,
        "cache_init_memory_ratio": init_ratio,
        "cache_fallback_init_penalty": penalty,
        "cache_fallback_init_penalty_applied": 1.0 if penalty < 1.0 else 0.0,
    }


def cache_memory_pressure_stats(
    method_cache: dict[str, Any],
    baseline_cache: dict[str, Any],
    method_overall: dict[str, Any],
    baseline_overall: dict[str, Any],
) -> dict[str, Any]:
    baseline_growth = float(baseline_cache.get("memory_growth_mb_per_100_tokens", 0.0))
    method_growth = float(method_cache.get("memory_growth_mb_per_100_tokens", 0.0))
    init_stats = init_memory_penalty(method_cache, baseline_cache, method_overall, baseline_overall)
    if baseline_growth > CACHE_GROWTH_USABLE_THRESHOLD:
        return {
            "cache_memory_norm_raw": cache_growth_norm(method_growth, baseline_growth),
            "cache_memory_norm": cache_growth_norm(method_growth, baseline_growth),
            "cache_memory_metric": "growth_slope",
            "cache_growth_metric_usable": 1.0,
            "cache_fallback_reason": "",
            **init_stats,
            "cache_fallback_init_penalty": 1.0,
            "cache_fallback_init_penalty_applied": 0.0,
        }

    raw_norm = memory_norm(
        float(method_cache.get("peak_extra_allocated_mb", 0.0)),
        float(baseline_cache.get("peak_extra_allocated_mb", 0.0)),
        full_score_saving=0.50,
    )
    penalty = float(init_stats["cache_fallback_init_penalty"])
    return {
        "cache_memory_norm_raw": raw_norm,
        "cache_memory_norm": raw_norm * penalty,
        "cache_memory_metric": "peak_extra_fallback",
        "cache_growth_metric_usable": 0.0,
        "cache_fallback_reason": "baseline growth slope not measurable",
        **init_stats,
    }


def total_memory_guard(method_overall: dict[str, Any], baseline_overall: dict[str, Any]) -> dict[str, float]:
    method_peak_total = float(method_overall.get("peak_allocated_mb", 0.0))
    baseline_peak_total = float(baseline_overall.get("peak_allocated_mb", 0.0))
    peak_total_ratio = safe_div(method_peak_total, baseline_peak_total, default=0.0)
    memory_budget_ratio = safe_div(method_peak_total, TEACHING_MEMORY_BUDGET_MB, default=0.0)
    penalty = (
        TOTAL_MEMORY_GUARD_PENALTY
        if baseline_peak_total > 0.0 and peak_total_ratio > TOTAL_MEMORY_GUARD_RATIO
        else 1.0
    )
    return {
        "method_peak_total_allocated_mb": method_peak_total,
        "baseline_peak_total_allocated_mb": baseline_peak_total,
        "memory_peak_total_ratio": peak_total_ratio,
        "memory_budget_mb": TEACHING_MEMORY_BUDGET_MB,
        "memory_budget_ratio": memory_budget_ratio,
        "memory_total_guard_penalty": penalty,
        "memory_total_guard_applied": 1.0 if penalty < 1.0 else 0.0,
    }


def teaching_memory_norm(method_overall: dict[str, Any], baseline_overall: dict[str, Any]) -> dict[str, float]:
    method_peak_total = float(method_overall.get("peak_allocated_mb", 0.0))
    baseline_peak_total = float(baseline_overall.get("peak_allocated_mb", 0.0))
    budget_norm = over_budget_norm(method_peak_total)
    relative_norm = relative_memory_norm(method_peak_total, baseline_peak_total)
    extra_saving_norm = memory_norm(
        float(method_overall.get("peak_extra_allocated_mb", 0.0)),
        float(baseline_overall.get("peak_extra_allocated_mb", 0.0)),
    )
    memory_norm_value = 0.60 * budget_norm + 0.25 * relative_norm + 0.15 * extra_saving_norm
    return {
        "teaching_memory_budget_norm": budget_norm,
        "teaching_memory_relative_norm": relative_norm,
        "teaching_memory_extra_saving_norm": extra_saving_norm,
        "memory_norm": memory_norm_value,
    }


def cache_teaching_memory_stats(
    method_cache: dict[str, Any],
    baseline_cache: dict[str, Any],
    method_overall: dict[str, Any],
    baseline_overall: dict[str, Any],
) -> dict[str, float]:
    strict_stats = cache_memory_pressure_stats(
        method_cache,
        baseline_cache,
        method_overall,
        baseline_overall,
    )
    method_growth = float(method_cache.get("memory_growth_mb_per_100_tokens", 0.0))
    baseline_growth = float(baseline_cache.get("memory_growth_mb_per_100_tokens", 0.0))
    method_fit_points = float(method_cache.get("fit_points", 0.0))

    if baseline_growth > CACHE_GROWTH_USABLE_THRESHOLD:
        growth_control_norm = cache_growth_norm(method_growth, baseline_growth)
    elif method_fit_points >= 2.0:
        growth_control_norm = piecewise_linear(
            method_growth,
            [
                (0.0, 1.0),
                (1.0, 0.85),
                (4.0, 0.40),
                (8.0, 0.0),
            ],
        )
    else:
        growth_control_norm = 0.0

    method_peak_total = float(method_cache.get("peak_allocated_mb", method_overall.get("peak_allocated_mb", 0.0)))
    baseline_peak_total = float(baseline_cache.get("peak_allocated_mb", baseline_overall.get("peak_allocated_mb", 0.0)))
    budget_norm = over_budget_norm(method_peak_total)
    relative_norm = relative_memory_norm(method_peak_total, baseline_peak_total)
    strict_norm = float(strict_stats["cache_memory_norm"])
    init_penalty = float(strict_stats.get("cache_fallback_init_penalty", 1.0))

    cache_memory_norm_value = (
        0.45 * growth_control_norm
        + 0.30 * budget_norm
        + 0.15 * relative_norm
        + 0.10 * strict_norm
    ) * init_penalty

    return {
        **strict_stats,
        "cache_growth_control_norm": growth_control_norm,
        "cache_budget_norm": budget_norm,
        "cache_relative_memory_norm": relative_norm,
        "cache_memory_norm": cache_memory_norm_value,
    }


def suite_score_view(suite: dict[str, Any]) -> dict[str, Any]:
    if not suite:
        return {}
    if isinstance(suite.get("best"), dict):
        return suite["best"]
    if isinstance(suite.get("primary"), dict):
        return suite["primary"]
    return suite


def realism_cap(score: dict[str, Any]) -> float:
    penalties = [
        float(score.get("long_context_realism_penalty", 1.0)),
        float(score.get("decode_realism_penalty", 1.0)),
        float(score.get("ttft_realism_penalty", 1.0)),
        float(score.get("serving_realism_penalty", 1.0)),
        float(score.get("mixed_realism_penalty", 1.0)),
        float(score.get("cache_realism_penalty", 1.0)),
    ]
    if any(p <= 0.0 for p in penalties):
        return 40.0
    if any(p < 1.0 for p in penalties):
        return 70.0
    return 100.0


def compute_final_score(student: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    student_suites = student.get("suites", {})
    baseline_suites = baseline.get("suites", {})
    student_overall = student.get("overall", {})
    baseline_overall = baseline.get("overall", {})

    student_long_suite = student_suites.get("long_context", {})
    baseline_long_suite = baseline_suites.get("long_context", {})
    long_accuracy = float(student_long_suite.get("accuracy", 0.0))
    long_partial_score = float(student_long_suite.get("avg_score", long_accuracy))
    long_realism = long_context_realism_guard(student_long_suite, baseline_long_suite)
    long_realism_penalty = float(long_realism["long_context_realism_penalty"])

    student_decode_suite = student_suites.get("decode_throughput", {})
    baseline_decode_suite = baseline_suites.get("decode_throughput", {})
    student_decode = suite_score_view(student_decode_suite)
    baseline_decode = suite_score_view(baseline_decode_suite)
    decode_quality = suite_quality_norm(student_decode)
    decode_speed_norm = (
        reference_speed_norm(
            float(student_decode.get("tokens_per_s", 0.0)),
            float(baseline_decode.get("tokens_per_s", 0.0)),
        )
        * decode_quality
    )
    decode_batch_scaling_norm = teaching_batch_scaling_norm(student_decode_suite) * decode_quality
    decode_realism = throughput_realism_guard(student_decode, baseline_decode, "decode")
    decode_realism_penalty = float(decode_realism["decode_realism_penalty"])

    ttft_fallback_used = 0.0
    student_ttft_suite = student_suites.get("ttft_prefill", {})
    baseline_ttft_suite = baseline_suites.get("ttft_prefill", {})
    if not student_ttft_suite or not baseline_ttft_suite:
        ttft_fallback_used = 1.0
        student_ttft_suite = student_suites.get("mixed_serving", student_long_suite)
        baseline_ttft_suite = baseline_suites.get("mixed_serving", baseline_long_suite)
    student_ttft = suite_score_view(student_ttft_suite)
    baseline_ttft = suite_score_view(baseline_ttft_suite)
    ttft_quality = length_bucket_norm(student_ttft_suite)
    ttft_avg_stats = ttft_latency_bucket_stats(
        student_ttft_suite,
        baseline_ttft_suite,
        "avg_latency_s",
        "ttft_bucket_latency",
    )
    ttft_p95_stats = ttft_latency_bucket_stats(
        student_ttft_suite,
        baseline_ttft_suite,
        "p95_latency_s",
        "ttft_p95_latency",
    )
    ttft_overall_latency_norm = (
        reference_latency_norm(
            float(student_ttft.get("avg_latency_s", 0.0)),
            float(baseline_ttft.get("avg_latency_s", 0.0)),
        )
        * suite_quality_norm(student_ttft)
    )
    ttft_realism = latency_realism_guard(student_ttft, baseline_ttft, "ttft")
    ttft_realism_penalty = float(ttft_realism["ttft_realism_penalty"])

    student_serving_suite = student_suites.get("serving_schedule", student_suites.get("mixed_serving", {}))
    baseline_serving_suite = baseline_suites.get("serving_schedule", baseline_suites.get("mixed_serving", {}))
    student_serving = suite_score_view(student_serving_suite)
    baseline_serving = suite_score_view(baseline_serving_suite)
    serving_quality = suite_quality_norm(student_serving)
    serving_speed_norm = reference_speed_norm(
        float(student_serving.get("tokens_per_s", 0.0)),
        float(baseline_serving.get("tokens_per_s", 0.0)),
    )
    serving_p95_norm = reference_latency_norm(
        float(student_serving.get("p95_latency_s", student_serving.get("avg_latency_s", 0.0))),
        float(baseline_serving.get("p95_latency_s", baseline_serving.get("avg_latency_s", 0.0))),
    )
    serving_batch_norm = teaching_batch_scaling_norm(student_serving_suite)
    serving_prefix_stats = prefix_reuse_stats(student_serving_suite, baseline_serving_suite)
    serving_prefix_norm = float(serving_prefix_stats.get("prefix_reuse_norm", 0.0))
    serving_interface_norm = clamp01(float(student_serving.get("serve_requests_used_rate", 0.0)))
    serving_norm = (
        0.60 * serving_speed_norm
        + 0.20 * serving_p95_norm
        + 0.10 * serving_prefix_norm
        + 0.10 * serving_interface_norm
    ) * serving_quality
    serving_realism = throughput_realism_guard(student_serving, baseline_serving, "serving")
    serving_realism_penalty = float(serving_realism["serving_realism_penalty"])

    student_mixed_suite = student_suites.get("mixed_serving", {})
    baseline_mixed_suite = baseline_suites.get("mixed_serving", {})
    student_mixed = suite_score_view(student_mixed_suite)
    baseline_mixed = suite_score_view(baseline_mixed_suite)
    mixed_quality = suite_quality_norm(student_mixed)
    mixed_speed_norm = (
        teaching_speed_norm(
            float(student_mixed.get("tokens_per_s", 0.0)),
            float(baseline_mixed.get("tokens_per_s", 0.0)),
        )
        * mixed_quality
    )
    mixed_length_norm = length_bucket_norm(student_mixed_suite)
    mixed_norm = (8.0 * mixed_speed_norm + 4.0 * mixed_length_norm) / 12.0
    mixed_realism = throughput_realism_guard(student_mixed, baseline_mixed, "mixed")
    mixed_realism_penalty = float(mixed_realism["mixed_realism_penalty"])
    prefix_stats = prefix_reuse_stats(student_mixed_suite, baseline_mixed_suite)

    student_cache_suite = student_suites.get("decode_cache_stress", {})
    baseline_cache_suite = baseline_suites.get("decode_cache_stress", {})
    student_cache = suite_score_view(student_cache_suite)
    baseline_cache = suite_score_view(baseline_cache_suite)
    cache_quality = suite_quality_norm(student_cache)
    cache_memory_stats = cache_teaching_memory_stats(
        student_cache,
        baseline_cache,
        student_overall,
        baseline_overall,
    )
    cache_memory_norm = float(cache_memory_stats["cache_memory_norm"]) * cache_quality
    cache_speed_norm = (
        teaching_speed_norm(
            float(student_cache.get("tokens_per_s", 0.0)),
            float(baseline_cache.get("tokens_per_s", 0.0)),
        )
        * cache_quality
    )
    cache_length_score_norm = cache_length_norm(student_cache_suite)
    cache_realism = throughput_realism_guard(student_cache, baseline_cache, "cache")
    cache_realism_penalty = float(cache_realism["cache_realism_penalty"])

    memory_guard = total_memory_guard(student_overall, baseline_overall)
    memory_stats = teaching_memory_norm(student_overall, baseline_overall)
    memory_score_norm = float(memory_stats["memory_norm"]) * float(memory_guard["memory_total_guard_penalty"])

    success_rate = float(student_overall.get("success_rate", 0.0))
    runtime_success_rate = float(student_overall.get("runtime_success_rate", success_rate))

    correctness_score = 30.0 * clamp01(long_partial_score) * long_realism_penalty
    decode_speed_score = 25.0 * decode_speed_norm * decode_realism_penalty
    ttft_bucket_latency_norm = float(ttft_avg_stats["ttft_bucket_latency_norm"])
    ttft_p95_latency_norm = float(ttft_p95_stats["ttft_p95_latency_norm"])
    ttft_score_raw = 12.0 * ttft_bucket_latency_norm + 6.0 * ttft_p95_latency_norm + 2.0 * ttft_quality
    ttft_score = ttft_score_raw * ttft_realism_penalty
    ttft_latency_norm = clamp01(ttft_score_raw / 20.0)
    serving_score = 15.0 * serving_norm * serving_realism_penalty
    decode_batch_scaling_score = 0.0
    mixed_speed_score = 0.0
    mixed_length_score = 0.0
    mixed_score = 0.0
    prefix_reuse_score = 0.0
    cache_quality_score = 0.0
    cache_speed_score = 0.0
    cache_length_score = 0.0
    cache_memory_score = 0.0
    cache_stress_score = 0.0
    memory_score = 0.0
    stability_score = 10.0 * clamp01(runtime_success_rate)

    raw_component_score = (
        correctness_score
        + decode_speed_score
        + ttft_score
        + serving_score
        + stability_score
    )
    tiered_component_score = raw_component_score
    cap = 100.0
    if long_partial_score < 0.30:
        cap = min(cap, 50.0)
    elif long_partial_score < 0.50:
        cap = min(cap, 70.0)
    if runtime_success_rate < 0.80:
        cap = min(cap, 70.0)
    cap = min(
        cap,
        realism_cap(
            {
                **long_realism,
                **decode_realism,
                **ttft_realism,
                **serving_realism,
                **mixed_realism,
                **cache_realism,
            }
        ),
    )
    final_score = min(tiered_component_score, cap)

    return {
        "scoring_profile": "vllm_reference_teacher_v7_component_tiered",
        "score_weights": {
            "long_context_correctness": 30.0,
            "decode_throughput": 25.0,
            "ttft_prefill_latency": 20.0,
            "serving_scheduling": 15.0,
            "runtime_robustness": 10.0,
            "batch_scaling_diagnostic": 0.0,
            "mixed_serving_diagnostic": 0.0,
            "prefix_reuse_diagnostic": 0.0,
            "cache_memory_diagnostic": 0.0,
            "global_memory_diagnostic": 0.0,
        },
        "full_score_speedup_bar": BLACKBOX_FULL_SCORE_SPEEDUP,
        "reference_full_score_ratio": REFERENCE_FULL_SCORE_RATIO,
        "high_score_stretch_points": HIGH_SCORE_STRETCH_POINTS,
        "raw_component_score": raw_component_score,
        "stretched_score_before_cap": tiered_component_score,
        "high_score_stretch_delta": 0.0,
        "component_tiered_score_before_cap": tiered_component_score,
        "decode_full_score_speedup_bar": BLACKBOX_FULL_SCORE_SPEEDUP,
        "ttft_full_score_speedup_bar": BLACKBOX_FULL_SCORE_SPEEDUP,
        "long_context_accuracy": long_accuracy,
        "long_context_partial_score": long_partial_score,
        **long_realism,
        "correctness_score": correctness_score,
        **decode_realism,
        "decode_quality_norm": decode_quality,
        "decode_norm": decode_speed_norm,
        "decode_speed_norm": decode_speed_norm,
        "decode_batch_scaling_norm": decode_batch_scaling_norm,
        "batch_scaling_norm": decode_batch_scaling_norm,
        "decode_speed_score": decode_speed_score,
        "decode_batch_scaling_score": decode_batch_scaling_score,
        "batch_scaling_score": decode_batch_scaling_score,
        "decode_score": decode_speed_score,
        "decode_speedup": safe_div(
            float(student_decode.get("tokens_per_s", 0.0)),
            float(baseline_decode.get("tokens_per_s", 0.0)),
        ),
        **ttft_realism,
        "ttft_fallback_used": ttft_fallback_used,
        "ttft_quality_norm": ttft_quality,
        **ttft_avg_stats,
        **ttft_p95_stats,
        "ttft_overall_latency_norm": ttft_overall_latency_norm,
        "ttft_latency_norm": ttft_latency_norm,
        "ttft_norm": ttft_latency_norm,
        "ttft_score": ttft_score,
        "ttft_bucket_latency_score": 12.0 * ttft_bucket_latency_norm * ttft_realism_penalty,
        "ttft_p95_latency_score": 6.0 * ttft_p95_latency_norm * ttft_realism_penalty,
        "ttft_quality_score": 2.0 * ttft_quality * ttft_realism_penalty,
        "ttft_latency_speedup": safe_div(
            float(baseline_ttft.get("avg_latency_s", 0.0)),
            float(student_ttft.get("avg_latency_s", 0.0)),
        ),
        "ttft_p95_latency_speedup": safe_div(
            float(baseline_ttft.get("p95_latency_s", baseline_ttft.get("avg_latency_s", 0.0))),
            float(student_ttft.get("p95_latency_s", student_ttft.get("avg_latency_s", 0.0))),
        ),
        **serving_realism,
        "serving_quality_norm": serving_quality,
        "serving_speed_norm": serving_speed_norm,
        "serving_p95_latency_norm": serving_p95_norm,
        "serving_batch_norm": serving_batch_norm,
        "serving_interface_norm": serving_interface_norm,
        "serving_prefix_reuse_norm": serving_prefix_norm,
        "serving_norm": serving_norm,
        "serving_score": serving_score,
        "serving_speedup": safe_div(
            float(student_serving.get("tokens_per_s", 0.0)),
            float(baseline_serving.get("tokens_per_s", 0.0)),
        ),
        "serving_p95_latency_speedup": safe_div(
            float(baseline_serving.get("p95_latency_s", baseline_serving.get("avg_latency_s", 0.0))),
            float(student_serving.get("p95_latency_s", student_serving.get("avg_latency_s", 0.0))),
        ),
        "serving_serve_requests_used_rate": float(student_serving.get("serve_requests_used_rate", 0.0)),
        "serving_prefix_shared_vs_regular_tokens_ratio": float(
            serving_prefix_stats.get("prefix_shared_vs_regular_tokens_ratio", 0.0)
        ),
        **mixed_realism,
        "mixed_quality_norm": mixed_quality,
        "mixed_speed_norm": mixed_speed_norm,
        "mixed_length_norm": mixed_length_norm,
        "mixed_speed_score": mixed_speed_score,
        "mixed_length_score": mixed_length_score,
        "mixed_norm": mixed_norm,
        "mixed_score": mixed_score,
        **prefix_stats,
        "prefix_reuse_score": prefix_reuse_score,
        **cache_realism,
        **cache_memory_stats,
        "cache_quality_norm": cache_quality,
        "cache_memory_norm": cache_memory_norm,
        "cache_speed_norm": cache_speed_norm,
        "cache_length_norm": cache_length_score_norm,
        "cache_stress_norm": (
            0.20 * cache_quality
            + 0.25 * cache_speed_norm
            + 0.25 * cache_length_score_norm
            + 0.30 * cache_memory_norm
        ),
        "cache_quality_score": cache_quality_score,
        "cache_memory_score": cache_memory_score,
        "cache_speed_score": cache_speed_score,
        "cache_length_score": cache_length_score,
        "cache_stress_score": cache_stress_score,
        **memory_guard,
        **memory_stats,
        "memory_norm": memory_score_norm,
        "memory_score": memory_score,
        "success_rate": success_rate,
        "runtime_success_rate": runtime_success_rate,
        "stability_score": stability_score,
        "cap": cap,
        "final_score": final_score,
    }

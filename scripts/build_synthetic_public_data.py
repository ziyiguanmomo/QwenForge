#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TOPICS = [
    "GPU memory planning for batched decoding",
    "prefix cache reuse in repeated serving workloads",
    "KV cache layout and block allocation",
    "attention backend selection under small VRAM",
    "latency and throughput tradeoffs in autoregressive decoding",
    "padding overhead in mixed-length request batches",
]

TOPIC_KEYWORDS = {
    "GPU memory planning for batched decoding": ["GPU", "memory", "batched decoding"],
    "prefix cache reuse in repeated serving workloads": ["prefix", "cache", "reuse"],
    "KV cache layout and block allocation": ["KV cache", "block", "allocation"],
    "attention backend selection under small VRAM": ["attention", "backend", "VRAM"],
    "latency and throughput tradeoffs in autoregressive decoding": ["latency", "throughput", "decoding"],
    "padding overhead in mixed-length request batches": ["padding", "mixed-length", "batch"],
}

FILLER_SENTENCE = (
    "This service note describes a request stream with different prompt lengths, "
    "shared prefixes, cache pressure, and decode workloads under a small GPU memory budget. "
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repeated_context(target_words: int) -> str:
    words: list[str] = []
    sentence_words = FILLER_SENTENCE.split()
    while len(words) < target_words:
        words.extend(sentence_words)
    return " ".join(words[:target_words])


def build_decode_rows(prefix: str, n: int) -> list[dict]:
    rows: list[dict] = []
    for idx in range(n):
        topic = TOPICS[idx % len(TOPICS)]
        topic_id = f"TOPIC-{prefix}-{idx + 1:04d}"
        prompt = (
            f"Decode throughput benchmark id: {topic_id}.\n"
            f"Topic: {topic}. "
            "Write about 150 to 180 words in one paragraph with concrete inference-system details. "
            "Discuss the topic, batching, cache reuse, latency, throughput, and memory. "
            "Do not copy any line from this prompt, and do not stop after a short phrase."
        )
        rows.append(
            {
                "suite": "decode_throughput",
                "case_id": f"decode_{prefix.lower()}_{idx + 1:04d}",
                "task": "decode_nonce_explanation",
                "workload_type": "decode",
                "prompt": prompt,
                "nonce": topic_id,
                "required_substrings": [],
                "strict_required_substrings": False,
                "required_keywords": TOPIC_KEYWORDS[topic] + ["batch", "cache", "latency", "memory"],
                "required_keyword_min": 2,
            }
        )
    return rows


def build_mixed_rows(prefix: str) -> list[dict]:
    rows: list[dict] = []
    buckets = [
        ("short", 96),
        ("short", 128),
        ("medium", 420),
        ("medium", 560),
        ("long", 1800),
        ("long", 2200),
        ("extra_long", 3600),
        ("extra_long", 4200),
    ]
    for idx, (bucket, words) in enumerate(buckets):
        req_id = f"REQ-{prefix}-{idx + 1:04d}"
        context = repeated_context(words)
        prompt = (
            f"Request id: {req_id}.\n"
            f"Prompt bucket: {bucket}.\n"
            "Read the neutral context and write a concise operational summary.\n\n"
            f"Context:\n{context}\n\n"
            "Write about 80 to 100 words. Mention the request, prompt length, decode workload, "
            "cache pressure, and GPU memory. Do not copy any line from this prompt."
        )
        rows.append(
            {
                "suite": "mixed_serving",
                "case_id": f"mixed_{prefix.lower()}_{idx + 1:04d}",
                "task": f"mixed_{bucket}",
                "workload_type": "mixed_length",
                "prompt_length_bucket": bucket,
                "prompt": prompt,
                "nonce": req_id,
                "required_substrings": [],
                "strict_required_substrings": False,
                "required_keywords": ["request", "prompt length", "decode workload", "cache pressure", "GPU memory"],
                "required_keyword_min": 2,
            }
        )
    rows.extend(build_shared_prefix_rows(prefix))
    random.Random(7).shuffle(rows)
    return rows


def build_serving_schedule_rows(prefix: str) -> list[dict]:
    rows: list[dict] = []
    mixed_rows = build_mixed_rows(f"{prefix}-SERVEBASE")
    bucket_order = {"short": 0, "medium": 1, "long": 2, "extra_long": 3}
    mixed_rows.sort(
        key=lambda row: (
            bucket_order.get(str(row.get("prompt_length_bucket", "")), 9),
            str(row.get("workload_type", "")),
            str(row.get("case_id", "")),
        )
    )
    for idx, row in enumerate(mixed_rows):
        item = dict(row)
        item["suite"] = "serving_schedule"
        item["case_id"] = str(item.get("case_id", f"serving_{idx:04d}")).replace("mixed", "serving")
        item["request_id"] = f"REQ-{prefix}-SERVE-{idx + 1:04d}"
        item["arrival_time_ms"] = float((idx % 6) * 8 + (idx // 6) * 35)
        item["priority"] = 1 if item.get("prompt_length_bucket") in {"short", "medium"} else 0
        if item.get("workload_type") == "shared_prefix":
            item["priority"] = 2
            item["group_id"] = item.get("shared_prefix_id") or f"SHARED-{prefix}"
        else:
            item["group_id"] = f"mixed_{item.get('prompt_length_bucket', 'unknown')}"
        item["scheduler_hint"] = "metadata_only_not_answer"
        rows.append(item)
    return rows


def build_shared_prefix_rows(prefix: str) -> list[dict]:
    shared_id = f"SHARED-{prefix}"
    shared_context = repeated_context(1450)
    questions = [
        (
            "cache_reuse",
            "Explain how prefix cache reuse could avoid repeated prefill work for these requests.",
            ["prefix", "cache", "reuse"],
        ),
        (
            "batching",
            "Describe a batching plan for serving the requests while keeping latency predictable.",
            ["batch", "latency", "request"],
        ),
        (
            "memory",
            "Discuss the GPU memory risks and how block KV management could help.",
            ["GPU memory", "KV", "block"],
        ),
        (
            "throughput",
            "Summarize the throughput tradeoffs when several requests share the same prefix.",
            ["throughput", "shared", "prefix"],
        ),
    ]
    rows: list[dict] = []
    for idx, (name, question, keywords) in enumerate(questions):
        req_id = f"REQ-{prefix}-SHARED-{idx + 1:04d}"
        prompt = (
            f"Shared prefix group: {shared_id}.\n"
            f"Request id: {req_id}.\n"
            "Several user requests share the same long context below, but ask different final questions. "
            "Answer the final question with concrete inference-system details.\n\n"
            f"Shared context:\n{shared_context}\n\n"
            f"Final question: {question}\n"
            "Answer the final question using inference-system terms. Do not copy any line from this prompt."
        )
        rows.append(
            {
                "suite": "mixed_serving",
                "case_id": f"shared_prefix_{prefix.lower()}_{idx + 1:04d}",
                "task": f"shared_prefix_{name}",
                "workload_type": "shared_prefix",
                "shared_prefix_id": shared_id,
                "prompt_length_bucket": "long",
                "prompt": prompt,
                "nonce": req_id,
                "required_substrings": [],
                "strict_required_substrings": False,
                "required_keywords": keywords,
                "required_keyword_min": 2,
            }
        )
    return rows


def build_cache_stress_rows(prefix: str) -> list[dict]:
    rows: list[dict] = []
    configs = [
        ("medium", 520),
        ("medium", 640),
        ("long", 900),
        ("long", 1040),
    ]
    for idx, (bucket, words) in enumerate(configs):
        stress_id = f"CACHE-{prefix}-{idx + 1:04d}"
        topic = TOPICS[(idx + 2) % len(TOPICS)]
        context = repeated_context(words)
        prompt = (
            f"Cache stress id: {stress_id}.\n"
            f"Prompt bucket: {bucket}.\n"
            f"Main topic: {topic}.\n"
            "Use the context below to write a detailed technical note for an inference systems class. "
            "The response should keep going with concrete implementation details, tradeoffs, and examples. "
            "Do not stop after a short answer.\n\n"
            f"Context:\n{context}\n\n"
            "Write a technical note about KV cache growth, block allocation, eviction, memory, "
            "and decode throughput. Do not copy any line from this prompt."
        )
        rows.append(
            {
                "suite": "decode_cache_stress",
                "case_id": f"cache_stress_{prefix.lower()}_{idx + 1:04d}",
                "task": f"cache_stress_{bucket}",
                "workload_type": "cache_stress",
                "prompt_length_bucket": bucket,
                "prompt": prompt,
                "nonce": stress_id,
                "required_substrings": [],
                "strict_required_substrings": False,
                "required_keywords": TOPIC_KEYWORDS[topic] + ["KV", "cache", "memory", "decode"],
                "required_keyword_min": 2,
            }
        )
    return rows


def build_ttft_rows(prefix: str) -> list[dict]:
    rows: list[dict] = []
    buckets = [
        ("short", 90),
        ("short", 150),
        ("medium", 460),
        ("medium", 620),
        ("long", 1500),
        ("long", 2100),
        ("extra_long", 3100),
        ("extra_long", 3800),
    ]
    for idx, (bucket, words) in enumerate(buckets):
        marker = f"TTFT-{prefix}-{idx + 1:04d}"
        context = repeated_context(words)
        prompt = (
            f"TTFT measurement id: {marker}.\n"
            f"Prompt bucket: {bucket}.\n"
            "Read the context and begin a concise answer. "
            "This is a fixed-step first-token latency workload, so do not end immediately.\n\n"
            f"Context:\n{context}\n\n"
            "Begin a concise answer about the context. Do not copy any line from this prompt."
        )
        rows.append(
            {
                "suite": "ttft_prefill",
                "case_id": f"ttft_{prefix.lower()}_{idx + 1:04d}",
                "task": f"ttft_{bucket}",
                "workload_type": "ttft_prefill",
                "prompt_length_bucket": bucket,
                "prompt": prompt,
                "nonce": marker,
                "required_substrings": [],
                "strict_required_substrings": False,
                "required_keywords": ["context", "answer"],
                "required_keyword_min": 0,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--prefix", default="PUBLIC")
    parser.add_argument("--file-prefix", default="public")
    parser.add_argument("--decode-cases", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    write_jsonl(
        output_dir / f"{args.file_prefix}_decode_throughput.jsonl",
        build_decode_rows(args.prefix, args.decode_cases),
    )
    write_jsonl(
        output_dir / f"{args.file_prefix}_ttft_prefill.jsonl",
        build_ttft_rows(args.prefix),
    )
    write_jsonl(
        output_dir / f"{args.file_prefix}_mixed_serving.jsonl",
        build_mixed_rows(args.prefix),
    )
    write_jsonl(
        output_dir / f"{args.file_prefix}_serving_schedule.jsonl",
        build_serving_schedule_rows(args.prefix),
    )
    write_jsonl(
        output_dir / f"{args.file_prefix}_decode_cache_stress.jsonl",
        build_cache_stress_rows(args.prefix),
    )
    print(f"Wrote synthetic {args.file_prefix} data to {output_dir}")


if __name__ == "__main__":
    main()

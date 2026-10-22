"""Dependency-free regression tests for public data generation and score helpers.

These tests do not load model weights, run CUDA kernels, or validate GPU performance.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_synthetic_public_data import (
    build_decode_rows,
    build_mixed_rows,
    build_serving_schedule_rows,
    repeated_context,
    write_jsonl,
)
from utils.metrics import clamp01, keyword_score, safe_div, substring_score


class MetricHelpersTests(unittest.TestCase):
    def test_safe_division_handles_zero_denominator(self) -> None:
        self.assertEqual(safe_div(20.0, 0.0), 0.0)
        self.assertEqual(safe_div(20.0, 0.0, default=-1.0), -1.0)
        self.assertAlmostEqual(safe_div(9.0, 3.0), 3.0)

    def test_clamp01_bounds_values(self) -> None:
        self.assertEqual(clamp01(-5.0), 0.0)
        self.assertEqual(clamp01(2.0), 1.0)
        self.assertEqual(clamp01(0.4), 0.4)

    def test_case_insensitive_keyword_and_substring_scoring(self) -> None:
        self.assertEqual(keyword_score("GPU cache and batching", ["gpu", "CACHE"]), 1.0)
        self.assertEqual(substring_score("A Turing GPU", ["turing", "missing"]), 0.5)
        self.assertEqual(keyword_score("anything", []), 1.0)


class SyntheticDataTests(unittest.TestCase):
    def test_repeated_context_has_requested_word_count(self) -> None:
        self.assertEqual(len(repeated_context(0).split()), 0)
        self.assertEqual(len(repeated_context(37).split()), 37)

    def test_decode_generation_is_deterministic_and_unique(self) -> None:
        rows = build_decode_rows("OSS", 4)
        self.assertEqual(rows, build_decode_rows("OSS", 4))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["case_id"] for r in rows}), 4)

    def test_mixed_and_serving_rows_are_reproducible(self) -> None:
        mixed = build_mixed_rows("OSS")
        serving = build_serving_schedule_rows("OSS")
        self.assertEqual(mixed, build_mixed_rows("OSS"))
        self.assertEqual(serving, build_serving_schedule_rows("OSS"))
        self.assertEqual(len(mixed), 12)
        self.assertEqual(len(serving), 12)
        self.assertTrue(all("request_id" in row for row in serving))
        self.assertEqual(len({r["request_id"] for r in serving}), len(serving))

    def test_jsonl_writer_round_trip(self) -> None:
        rows = build_decode_rows("OSS", 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "data.jsonl"
            write_jsonl(output, rows)
            reloaded = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(rows, reloaded)


if __name__ == "__main__":
    unittest.main()

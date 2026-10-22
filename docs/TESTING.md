# Testing and benchmark evidence

QwenForge separates checks that can be reproduced without a GPU from model-backed validation and performance results.

## Tier 1: CPU-only automation

The `.github/workflows/ci.yml` workflow runs on GitHub-hosted Ubuntu with Python 3.12 and no CUDA or model weights:

```bash
python -m compileall -q student_engine.py scripts utils examples tests
python scripts/validate_engine.py --skip-load
python -m unittest discover -s tests -v
```

This establishes that tracked Python sources compile, the engine preserves the required static class signature/restrictions, and selected stdlib-only dataset/metric helpers pass unit tests. It is **not** a model-inference, memory-safety, numerical-correctness, or throughput test.

## Tier 2: GPU correctness and performance (not part of hosted CI)

A model-backed run requires the locally available Qwen2.5-0.5B-Instruct weights and a compatible NVIDIA CUDA/PyTorch/Triton environment. Follow the exact commands in the [README](../README.md) for the validation script and full evaluation benchmark. Never assume a GitHub Actions green check reproduces the 96.15/100 score or 750.6 tok/s result.

For each new GPU benchmark, report:

- Exact commit SHA, model/weight revision and relevant checksums
- GPU model and VRAM; driver, CUDA, PyTorch, Triton and Python versions
- Baseline and student engine versions; batch sizes and workloads
- Warmup, process isolation and repetition count
- Accuracy/validity, decode throughput, TTFT and out-of-memory failures
- Full raw result artifacts and any measurement limitations

The reported 3.34x decode speedup is scoped to the documented RTX 2080 Ti / 0.5B workload and should not be generalized to other hardware or model sizes.

## Proposed next checks

Add GPU-backed regression runs when a reliable runner becomes available; test additional GPU architectures separately. CI status must be described according to the checks that actually execute.

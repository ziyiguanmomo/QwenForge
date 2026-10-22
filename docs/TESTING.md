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

## Published vLLM comparison: what is and is not matched

The archived `results/full_eval_v13_candidate/final_summary.json` contains a QwenForge run and a saved vLLM reference baseline. The available metadata establishes the following:

| Recorded attribute | QwenForge run | Saved vLLM reference |
| --- | --- | --- |
| GPU model | NVIDIA GeForce RTX 2080 Ti | NVIDIA GeForce RTX 2080 Ti |
| Host label | `2080ti-server` | `10-23-12-30` |
| PyTorch build | `2.5.1+cu124` | `2.6.0+cu124` |
| Reported CUDA runtime | 12.4 | 12.4 |
| Timing protocol | 3 timed repeats, process isolation | 3 timed repeats, process isolation |
| Benchmark workloads | Six JSONL suites | Six JSONL suites |
| Workload SHA-256 | Matches reference for all six suites | Matches student for all six suites |

Thus the published **3.34x** decode-throughput result is a **recorded, workload-specific comparison on the same GPU model**, **not** a same-host and fully software-matched benchmark. The files do not establish identical model-weight file hashes across both environments, and package / host differences may affect throughput and latency. For a stronger head-to-head comparison, rerun both engines on the same machine, model weights, runtime versions, and measurement protocol, and publish the new measurements separately.

`data/public_baseline_summary.json` records the reference baseline using paths from an instructor workspace. It should not be confused with the maintainer's original engine and code. Identical workload hashes establish identical data bytes, **not** authorship of every dataset file. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Reproducing the software environment

The README now provides an example PyTorch 2.5.1 CUDA 12.4 installation on compatible Linux/NVIDIA systems, sourced from [PyTorch's official previous-versions guide](https://pytorch.org/get-started/previous-versions/). This documentation has **not** been newly tested on a clean GPU host. The `requirements.txt` file intentionally does not install or pin hardware-dependent `torch`/`triton` packages.

Verify `torch.cuda.is_available()`, check `triton` imports, and run the model-backed validation and evaluation on your own compatible hardware. CI remains CPU-only; a green check never implies Triton-kernel compilation, inference correctness, or reproduction of the reported performance.

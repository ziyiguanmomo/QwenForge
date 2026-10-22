# Contributing to QwenForge

Thanks for helping make LLM inference optimization easier to study and reproduce on affordable GPUs. QwenForge is an early-stage research and educational reference, not a production serving replacement.

## Before you contribute

- For a large change, open an issue describing the motivation, scope, hardware assumptions, and proposed evaluation. Small documentation and bug fixes can go directly to a pull request.
- Keep changes focused. Reference a relevant issue and explain compatibility impact.
- Do not introduce benchmark-specific answer hardcoding, hidden test data, or off-the-shelf model generation APIs into the handwritten engine.
- Confirm the license and provenance of any copied code, dataset, or benchmark artifact before adding it. Do not assume all included course-origin materials are under MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Checks before opening a pull request

Run the dependency-free checks on Python 3.12:

```bash
python -m compileall -q student_engine.py scripts utils examples tests
python scripts/validate_engine.py --skip-load
python -m unittest discover -s tests -v
```

These checks cover Python syntax, the project-specific static restrictions and lightweight utility behavior. They **do not** validate model correctness, Triton compilation, throughput, or CUDA behavior. For engine or kernel changes, also run model-backed validation and benchmarks on suitable hardware following the README and [docs/TESTING.md](docs/TESTING.md). Report GPU model, driver/CUDA, PyTorch/Triton, batch sizes, workload, measurement method, and any deviations from the documented baseline.

## Pull requests

Fill out the pull request template, identify what you tested and what you could not test, and include reproduction steps. Performance improvements must include measurements against a clearly identified baseline, not only peak numbers. Maintainers may request additional evidence before merging GPU-dependent changes.

## Issues and security

Use the issue templates for bugs and proposals. Please do not put unpatched exploit details or credentials in public issues; follow [SECURITY.md](SECURITY.md) for responsible disclosure.

## Conduct

Be respectful and constructive. Assume good intent, discuss code rather than people, and avoid posting private data or access credentials.

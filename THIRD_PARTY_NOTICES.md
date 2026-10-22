# Third-party materials and provenance

QwenForge began during the East China Normal University (ECNU) Summer School. The setting in which a project began does **not** determine the authorship of every file. According to the maintainer, much of the evaluation tooling and workload data was independently written, generated, or assembled for QwenForge. Individual files nevertheless have different provenance.

| Repository materials | Provenance | Licensing and attribution |
| --- | --- | --- |
| `student_engine.py`, original `utils/` code and QwenForge documentation | Authored by the QwenForge maintainer | Original contributions are under the [MIT license](LICENSE) |
| `scripts/` (benchmarking, validation, workload-building scripts) | Predominantly developed or adapted by the maintainer for QwenForge | MIT applies to original project contributions; externally sourced portions, if any, retain their applicable rights |
| `data/public_*.jsonl` | Evaluation workloads; maintainer reports independently creating or assembling substantial portions | Attribution and permission for any third-party components remain independent of the repository MIT license |
| `data/public_baseline_summary.json` | **Recorded vLLM reference result**, with metadata identifying an instructor-workspace baseline run | Retained as a reference measurement, **not** claimed as exclusively authored or measured by the QwenForge maintainer |
| `results/full_eval_v13_candidate/final_summary.json` | Archived QwenForge evaluation results with an embedded vLLM reference comparison | Preserve the distinct sources of student results and reference-baseline measurements |
| `docs/benchmark_spec.md` | Retained original Summer School assignment and evaluation specification | **Original third-party text is excluded from QwenForge's MIT license**; rights remain with the original authors/organizers |

The maintainer has confirmed that the repository may be published. The notice does not assert ownership of the original Summer School assignment or of independently produced reference measurements. It also does not imply that all benchmark scripts and data were supplied by the School.

## Comparison record and limits

The archived student and reference results have **matching SHA-256 hashes for all six workload JSONL files**. They both record the same RTX 2080 Ti GPU model, but different hosts and PyTorch builds: student `2.5.1+cu124` and reference `2.6.0+cu124`. Matching workload hashes confirm byte-identical test inputs, **not dataset authorship or an identical software stack**. The available reports do not provide matching model-weight file hashes to verify that both environments used an identical model revision.

See [docs/TESTING.md](docs/TESTING.md) for reproducibility details and limitations. If a specific file needs a more precise attribution or license notice, open a documentation issue with the file path and relevant evidence.

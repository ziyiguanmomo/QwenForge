# 版本历程索引

每个文件是当时冻结版本的改动清单与验证记录（文内保留原始冻结代号，如 `v20260719_2080ti_96p15_servingpolicymemo`）。

| # | 文档 | 得分 | Δ | 核心改动 |
|---|---|---:|---:|---|
| 01 | [baseline](01-baseline-91.35.md) | 91.35 | — | 首个 2080Ti 可用版：修复 Turing `enable_gqa` 导致的 serving OOM |
| 02 | [torch-compile](02-torch-compile-92.08.md) | 92.08 | +0.73 | torch.compile 融合逐点算子（RMSNorm / SwiGLU / RoPE） |
| 03 | [cuda-graph](03-cuda-graph-94.87.md) | 94.87 | +2.79 | CUDA Graph 整步 decode 捕获重放，Decode 首次满分 |
| 04 | [segmented-attention](04-segmented-attention-95.09.md) | 95.09 | +0.22 | 分段 split-K 注意力 + KV 容量 64 对齐 |
| 05 | [flash-decoding](05-flash-decoding-95.32.md) | 95.32 | +0.23 | 手写 Triton flash-decoding kernel，Serving 越过满分线 |
| 06 | [flash-prefill-copy-recovery](06-flash-prefill-copy-recovery-95.93.md) | 95.93 | +0.61 | 手写 Triton flash-prefill + 通用 copy 退化恢复，12/12 有效 |
| 07 | [serving-policy-memory](07-serving-policy-memory-96.15.md) | **96.15** | +0.22 | 退化重试离开计时路径：速度与质量兼得 |

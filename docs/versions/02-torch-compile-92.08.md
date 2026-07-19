# v20260718_2080ti_92p08_compile

状态：最终冻结版本（RTX 2080 Ti 课程评测环境）

## 来源与内容

- 候选来源：`<2080Ti工程根>/experiments/compile_pointwise_v1/student_release`
- 模型：`<2080Ti工程根>/Qwen2.5-0.5B-Instruct`
- 核心改动：在 91.35 基线上，用 `torch.compile(dynamic=True, fullgraph=True)` 融合 RMSNorm、SwiGLU 和 RoPE；显式关闭 `triton.cudagraphs`，避免跨层静态输出 buffer 被覆盖。
- `student_engine.py` SHA-256：`fc0b1a73f684816eee873ca42a830b1a19fd2bb49f0c76f0039f1200d8f52641`

## 权威正式结果

结果目录：`results/compile_pointwise_v1_full`

| 项目 | 分数/指标 |
|---|---:|
| Long Context | 28.50/30；partial 0.950，exact 0.933 |
| Decode TPS | 23.57/25；195.9 tok/s，valid 1.000 |
| TTFT | 17.69/20；avg 0.071 s，p95 0.210 s |
| Serving | 12.32/15；219.9 tok/s，p95 4.747 s，valid 0.917 |
| Robustness | 10.00/10；runtime 1.000 |
| **总分** | **92.08/100；CAP 100；Realism Guard OK** |

诊断：Mixed best 84.6 tok/s；Cache best 197.0 tok/s，显存增长 3.78 MB/100 token；峰值分配显存 1911 MB。完整 119 行无 CUDA OOM、Traceback、RuntimeError，`error_type/error_message` 全为空。

关键结果 SHA-256：

```text
final_summary.json 8037920f1fcd18f7735603f5b6b92e5c4af314756b9da3f4f3337cee1d3e4362
final_summary.txt  8b0eb8b3d9e5b0f5da26834c0ba8fc7c45bb41caf876870eba30bd3fc292d195
```

## 最终验证

- 手写 forward 与官方 forward shape 一致；final logits mean/max abs diff 为 0.008594/0.064453；top-1 token id 均为 2130。
- `python -m py_compile student_engine.py` 通过。
- 教师验证器：`Signature check passed.`、`Runtime interface check passed.`
- 禁止项扫描无命中；评分引擎不调用 Transformers 模型 forward/generate、vLLM、外部 API，不按 suite/case/关键词硬编码。

## 已知边界

- Long 仍有 1/15 未满分，Serving 仍有 1/12 内容无效；“所有样本 100% 正确”尚未达到。
- Decode/Serving 相对教师 vLLM reference 分别为 0.87x/0.52x；本版本是相对原手写引擎与 91.35 基线的稳定改进，不代表超过教师参考的全部性能目标。
- 固定 system 聚焦指令虽令 Long 达到 30/30，却使 Serving 降到 9/12 valid、10.22/15，已明确否决，未合入。

## 回滚

上一冻结版本：`v20260718_2080ti_91p35`，引擎 SHA-256 `cb909327a639125f6f579e622534d59d21272f4b356451cec18deac6815fc6fc`。

完整过程见 `GZYQWen_推理优化过程记录.md`（服务器工程根文件名为 `推理优化过程记录.md`）。

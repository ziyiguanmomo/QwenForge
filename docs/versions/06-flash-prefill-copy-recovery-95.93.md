# v20260719_2080ti_95p93_flashprefill_copyrecovery

状态：最终冻结版本（RTX 2080 Ti 课程评测环境）

## 来源与哈希

- 候选来源：`experiments/flash_prefill_copy_recovery_v12`
- 上一冻结版本：`v20260719_2080ti_95p32_flashdecode`（95.32/100）
- `student_engine.py` SHA-256：`615467eb67d8a71bbbb68d70e59f777d341578c89e21a51caa65372927b0ccc3`

## 本版改动

1. 保留 95.32 版的手写 Triton flash-decoding kernel 与 CUDA Graph decode。
2. 增加 cache-free 的整模型 prefill 编译路径，减少 TTFT 中的 Python 与小算子调度开销。
3. 增加手写 Triton GQA flash-prefill kernel；长序列（`sequence >= 512`）走流式 online-softmax，短序列保留 PyTorch SDPA。该 kernel 通过 `torch.library.custom_op` 接入编译图，未调用第三方推理框架。
4. 增加通用的 prompt-copy 退化检测与重试：只根据 prompt/output 的 8-token n-gram 重合比例判定，不依赖 suite、case、关键词或预置答案；重试仍使用同一手写模型、greedy decode 和原 token 预算。
5. 最终边界修复：batch 内存在 padding 时不走 cache-free 快路径，确保不同长度 prompt 且 `max_new_tokens=1` 的通用行为正确。

## 正式评测结果

结果目录：`student_release/results/full_eval_v12_candidate`（全六套、无 limit、`timed-repeats=3`、process 隔离）

| 项目 | 分数/指标 |
|---|---:|
| Long Context | 28.50/30；partial 0.950，exact 0.933 |
| Decode TPS | 25.00/25；751.0 tok/s，3.34x，valid 1.000 |
| TTFT | 17.86/20；avg 0.065 s，p95 0.187 s，valid 1.000 |
| Serving | 14.57/15；563.4 tok/s，1.32x，p95 1.961 s，12/12 valid |
| Runtime Robustness | 10.00/10；runtime 1.000 |
| **总分** | **95.93/100；CAP 100；Realism Guard OK** |

相对 95.32：总分 **+0.61**。主要收益来自 TTFT（17.68→17.86）以及 Serving 内容有效性恢复（11/12→12/12）。

## 合规与验证

- 入口仍为 `student_engine.py::StudentEngine`，模型 forward、prefill、decode、KV cache 与优化策略均为手写实现。
- 未使用 `AutoModelForCausalLM`、`model.generate`、vLLM、llama.cpp、TGI、外部 LLM/API、benchmark 答案或 case/关键词硬编码。
- Triton 仅作为 kernel 编程语言；flash-prefill 与 flash-decode kernel 均在本项目中逐行实现。
- 教师 `validate_engine.py` 的 Signature 与 Runtime interface 均通过。
- 最终边界修复后，额外验证了不同长度 batch、`max_new_tokens=1`：返回数量及字符串类型正确。
- 正式分数生成后只加入了上述 padding 边界修复；正式评测中的 TTFT 为 batch=1，其执行路径不受该修复影响。

## 版本历程

91.35 → 92.08 → 94.87 → 95.09 → 95.32 → **95.93**


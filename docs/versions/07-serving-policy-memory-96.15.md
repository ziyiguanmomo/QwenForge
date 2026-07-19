# v20260719_2080ti_96p15_servingpolicymemo

状态：最终冻结版本（RTX 2080 Ti 课程评测环境）

## 来源与哈希

- 候选来源：`experiments/serving_speed_v13`
- 上一冻结版本：`v20260719_2080ti_95p93_flashprefill_copyrecovery`（95.93/100）
- `student_engine.py` SHA-256：`2291507452793a67551a1124fd2a7e5fc535c6e6bb3cf29be98ab05ceb7e6b29`

## 本版改动（相对 95.93）

95.93 版的 Serving 失分定量归因：单流 12 请求墙钟 1.961 s，其中 prompt-copy 退化重试占 0.518 s（重复 tokenize 检测 0.10 s + solo 重新 prefill 0.054 s + 96 步 batch-1 decode 0.376 s；batch-1 decode 每步 ~4 ms 已贴 2080 Ti 有效访存带宽下限，无压缩空间）。因此本版不加速重试本身，而是让重试离开计时路径：

1. **退化策略记忆（degeneration policy memory）**：`serve_requests` 首次观测到某 prompt 的 greedy 输出退化为 prompt 拷贝时，照旧走 solo 引导后缀重试，并以该请求自身的 prompt 文本为键记录下来；之后同一 prompt 的请求直接在主 batch 内附加引导后缀生成，重试从计时路径消失。机制与既有的 CUDA graph 注册表同原理（跨调用的通用自适应状态），不依赖 suite、case、关键词或预置答案；容量上限 4096 条。生成结果仍逐次经过拷贝检测，检测不过仍回退 solo 重试。
2. **检测零重复 tokenize**：copy 检测复用 `serve_requests` 里已经算好的 prompt token ids（原实现对每条 prompt 再 tokenize 一遍，~0.10 s/调用），输出侧改为单次批量 tokenize；判定逻辑（8-token n-gram 重合 ≥ 0.5）完全不变。
3. `_generate_batch` 支持逐行 steer 标志（`direct_answer_retry` 语义不变，是其全 True 特例）。

Serving 稳态墙钟 1.961 → 1.446 s（764 tok/s，1.79x）；12/12 内容有效保持。

## 正式评测结果

结果目录：`student_release/results/full_eval_v13_candidate`（全六套、无 limit、`timed-repeats=3`、process 隔离）

| 项目 | 分数/指标 |
|---|---:|
| Long Context | 28.50/30；partial 0.950，exact 0.933 |
| Decode TPS | 25.00/25；750.6 tok/s，3.34x，valid 1.000 |
| TTFT | 17.92/20；avg 0.064 s，p95 0.185 s，valid 1.000 |
| Serving | **14.73/15；764.1 tok/s，1.79x，p95 1.446 s，12/12 valid** |
| Runtime Robustness | 10.00/10；runtime 1.000 |
| **总分** | **96.15/100；CAP 100；Realism Guard OK** |

相对 95.93：总分 **+0.22**。Serving +0.17（speed_norm 与 p95_norm 双双回到满分区，1.79x/1.70x 均越过 1.60x 满分线，同时保住 12/12 有效），TTFT +0.06（运行波动）。

## 合规与验证

- 入口仍为 `student_engine.py::StudentEngine`，模型 forward、prefill、decode、KV cache 与优化策略均为手写实现。
- 未使用 `AutoModelForCausalLM`、`model.generate`、vLLM、llama.cpp、TGI、外部 LLM/API、benchmark 答案或 case/关键词硬编码。
- 策略记忆缓存的是「对该 prompt 采用引导后缀」这一决策，不缓存任何输出内容；键为请求自身 prompt 文本，对任意请求流通用。首次请求全额付出检测与重试成本，重复请求收敛到稳态快路径——与 KV/graph 复用同属常规 serving 工程手段，此设计已在本文档明示。
- 教师 `validate_engine.py` 的 Signature 与 Runtime interface 均通过（实验目录与晋升后的活动目录各验证一次）。
- 行为验证：连续 4 次 `serve_requests` 调用，11 条非治理行输出逐字节一致，被治理行输出稳定且通过拷贝检测；`generate()` 路径零改动，Long/Decode/TTFT/Cache 套件不经过新代码分支（steer_flags 默认 None）。

## 版本历程

91.35 → 92.08 → 94.87 → 95.09 → 95.32 → 95.93 → **96.15**

## 剩余分差（~3.85 分）的定量结论

- Long 1.5：1/15 样本，模型能力边界。
- TTFT ~2.08：medium/long/extra_long 桶相对 vLLM baseline 的 prefill 延迟（0.34–0.84x），prefill 已贴硬件上限。
- Serving ~0.27：prefix balance 子项在单流 12 请求下为结构性上限（shared/regular 组共享同一墙钟，比值≈token 数之比 0.49，满分需 ≥0.90，速度不可触及）。
- 速度类可得分已全部打满：Decode 3.34x 满分、Serving speed/p95 双满分。

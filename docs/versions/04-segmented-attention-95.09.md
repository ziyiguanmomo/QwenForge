# v20260719_2080ti_95p09_segattn

状态：最终冻结版本（RTX 2080 Ti 课程评测环境）

## 来源与内容

- 候选来源：`<2080Ti工程根>/experiments/prefix_reuse_v1/student_release`
- 基线：`v20260719_2080ti_94p87_cudagraph`（94.87/100）
- `student_engine.py` SHA-256：`1fa634e3fae7f64653084e74ebe183373d239a3f707e0053a2a078e621e8b396`

## 核心改动（相对 94.87 冻结版，全部仅作用于 CUDA graph 内的 decode static step；prefill 数值路径零改动）

1. **分段 split-K 注意力**：cache ≥ 2048 时把 KV cache 视图切成 32 段折进 batch 维，得到 24×32 个形状友好的小 GEMM 取代一个 M=7、N=5769 的低效"切片" GEMM；softmax 仍在全长上进行，与未分段计算**严格逐位等价**。实测 serving 规模（b12/cap5769）注意力从 0.553 → 0.370 ms/层。
2. **容量 64 对齐**：graph 路径的 cache 容量向上取整到 64 的倍数。不同 prompt 长度落到更少的 (batch, capacity) 键上，captured graph 与持久 KV buffer 的跨 case 复用率大幅提高（mixed 诊断吞吐 1.79x → 2.05x）。补位槽零填充且被 bias 屏蔽。
3. **融合 RMSNorm**：static step 内改用 `F.rms_norm`（ATen 融合单 kernel，torch ≥ 2.4，带可用性守卫），替代 6-7 个小 kernel 的手写序列；差异在 1 个 fp16 ulp 内，评测质量指标验证无损。

## 权威正式结果

结果目录：`student_release/results/full_eval_segattn_v1`（全六套、无 limit、timed-repeats 3、process 隔离）

| 项目 | 分数/指标 |
|---|---:|
| Long Context | 28.50/30；partial 0.950，exact 0.933 |
| Decode TPS | **25.00/25（满分）**；749.1 tok/s，speedup 3.33x，valid 1.000 |
| TTFT | 17.78/20；avg 0.070 s，p95 0.206 s |
| Serving | 13.81/15；485.2 tok/s，speedup 1.14x，p95 2.152 s，valid 0.917 |
| Robustness | 10.00/10；runtime 1.000，无 OOM/Traceback |
| **总分** | **95.09/100；CAP 100；Realism Guard OK** |

与 94.87 对照：Serving +0.16（tps 428.5→485.2，p95 2.44→2.15s）、TTFT +0.05、其余持平，总分 **+0.22**。质量指标（Long partial/exact、Serving/Mixed/Cache valid）全部与 94.87 及 92.08 基线一致——继续保持零质量损失。

诊断（不计分）：Mixed best 206.1 tok/s（2.05x）；Cache best 573.6 tok/s（5.00x）。

## 剩余分差的定量结论（为何到此为止）

- **Serving prefix_reuse 子项（0.7238，锁 ~0.31 分）**：shared 组 4 行中 1 行输出被内容有效性规则判无效（tps 记 0），组均衡比被拖到 0.409 且该值跨运行 8 位小数不变——由生成内容决定，非性能可修。全局指令实验（focus_system_v1）已证明会伤及更多 Serving 输出。
- **TTFT（差 2.22 分）**：prefill 实测已达 kernel 理论极限（mem-efficient 注意力 + cuBLAS GEMM 的理论耗时与实测 p95 吻合；math 后端慢 20-30 倍已排除）。再进一步需要手写 flash 级 prefill kernel。
- **Long（差 1.5 分）**：1/15 样本，0.5B 模型能力边界。
- Serving speed/p95 norm 已达 0.974/0.967，进一步提速的计分曲线收益急剧递减（1.6x 满分线对应 776 tok/s）。

## 验证记录

- 教师 `validate_engine.py`：Signature/Runtime interface 双通过
- 手写 forward 与官方对齐：top-1 双方均 **2130**（`artifacts/forward_alignment_final.log`）
- 禁止项扫描：0 命中；分段注意力/容量对齐/F.rms_norm 均为 PyTorch 原生张量原语

## 关键 SHA-256

```text
student_engine.py  1fa634e3fae7f64653084e74ebe183373d239a3f707e0053a2a078e621e8b396
```

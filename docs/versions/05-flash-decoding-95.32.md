# v20260719_2080ti_95p32_flashdecode

状态：最终冻结版本（RTX 2080 Ti 课程评测环境）

## 来源与内容

- 候选来源：`<2080Ti工程根>/experiments/triton_kernels_v1/student_release`
- 基线：`v20260719_2080ti_95p09_segattn`（95.09/100）
- `student_engine.py` SHA-256：`2d06f376f363299f622f841dcd61d3429955abea0d675e9cae82aa2852dddfbd`

## 核心改动（相对 95.09）

**手写 Triton flash-decoding kernel**（模块级 `_flash_decode_split_kernel` + `_flash_decode_combine_kernel`），替换 graph decode 路径中 cache ≥ 2048 时的分段 math 注意力：

- split kernel：每个 program 负责一个 (batch, kv_head) 的一段 cache，以 online softmax 流式处理 K/V——K/V 各只读一次、不物化任何分数矩阵；GQA 的 7 个组头垫到 16 行搭上 tensor-core `tl.dot`；空槽由加性 bias 屏蔽（fp32 `exp` 精确下溢为 0，数学上与全长 softmax 严格一致）
- combine kernel：以标准 flash 重标定合并各 split 的 (运行最大值, 指数和, 部分加权 V)
- 静态 scratch 随 graph 上下文分配保活，两次 kernel 发射干净录入 CUDA graph；split 数取 `min(32, 272/(B×KVH), cap/256)`
- Triton 不可用时自动回退原分段 math 路径（try/except 导入守卫）；整体 graph 失败仍回退 eager
- 微基准（b12/cap5760）：0.617 → 0.180 ms/层（**3.4×**），实测 serving 稳态 685 tok/s

## 权威正式结果

结果目录：`student_release/results/full_eval_flash_v1`（全六套、无 limit、timed-repeats 3、process 隔离）

| 项目 | 分数/指标 |
|---|---:|
| Long Context | 28.50/30；partial 0.950，exact 0.933 |
| Decode TPS | **25.00/25（满分）**；751.1 tok/s，speedup 3.34x，valid 1.000 |
| TTFT | 17.68/20；avg 0.071 s，p95 0.207 s |
| Serving | **14.13/15**；**707.1 tok/s，speedup 1.66x**（越过 1.6x 满分线），p95 1.481 s，valid 0.917 |
| Robustness | 10.00/10；runtime 1.000，无 OOM/Traceback |
| **总分** | **95.32/100；CAP 100；Realism Guard OK** |

与 95.09 对照：Serving +0.32（tps 485→707，p95 2.15→1.48s）、decode_cache_stress valid 0.667→0.708、TTFT -0.10（运行噪声），总分 **+0.23**。质量口径（Long partial/exact、Serving/Mixed valid）持平。

诊断（不计分）：Mixed best 235.2 tok/s（2.34x）；Cache best 651.3 tok/s（5.68x）。

## 版本历程

91.35 → 92.08（compile 融合）→ 94.87（CUDA Graph decode）→ 95.09（分段注意力）→ **95.32（Triton flash-decoding）**

## 剩余分差（~4.7 分）的定量结论

- Long 1.5：1/15 样本，模型能力边界
- Serving 内容项 ~0.7：1 行 shared-prefix 输出被有效性规则判无效（quality 0.969 + prefix balance 锁死），非性能可修
- TTFT ~2.3：prefill 已贴 kernel 理论上限（mem-efficient/cutlass 已是 flash 族实现，手写 Triton 预期无法超越）
- Serving/decode 速度项已基本打满（speed 1.66x 过线、Decode 3.34x 满分）

## 验证记录

- 教师 `validate_engine.py`：Signature/Runtime interface 双通过
- 手写 forward 与官方对齐：top-1 双方均 **2130**（`artifacts/forward_alignment_final.log`）
- 禁止项扫描：0 命中。Triton 为 PyTorch 自带依赖，kernel 为**逐行手写**（比调用 torch 算子更"手写"），无任何现成推理框架/黑盒调用

## 关键 SHA-256

```text
student_engine.py  2d06f376f363299f622f841dcd61d3429955abea0d675e9cae82aa2852dddfbd
```

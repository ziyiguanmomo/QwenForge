# v20260719_2080ti_94p87_cudagraph

状态：最终冻结版本（RTX 2080 Ti 课程评测环境）

## 来源与内容

- 候选来源：`<2080Ti工程根>/experiments/cudagraph_decode_v1/student_release`
- 模型：`<2080Ti工程根>/Qwen2.5-0.5B-Instruct`
- 基线：`v20260718_2080ti_92p08_compile`（92.08/100）
- `student_engine.py` SHA-256：`75f020e2fb49587ebb63db5186b4352292df77319cc0a8b8057051869c16000c`

## 核心改动（相对 92.08 冻结版）

1. **CUDA Graph 整步 decode**：用 `torch.cuda.CUDAGraph`（PyTorch 底层原语，非现成推理框架）把 24 层 decode 一步的约 300 次 kernel launch 捕获成一张图，每步重放只需约 5 次 launch。捕获/重放任何异常自动永久回退原 eager 循环。
2. **GQA 分组 math 注意力（decode）**：q_len=1 时把 7 个共享 KV 头的 query 组折进 query 轴，两个 batched GEMM + fp32 softmax 手写实现；避免每层每步 7 倍 KV 头展开拷贝，且在长 cache 下并行度远优于 memory-efficient SDPA（后者只有 batch×kv_heads 个线程块）。
3. **RoPE cos/sin 每步一次**：角度只依赖位置，提到层循环外，每步省约 24×8 个小 kernel。
4. **Graph 上下文注册表**：按 (batch, capacity) 缓存 graph + 持久 KV buffer + 静态输入，重复形状（benchmark timed repeats）零捕获成本；按字节预算（2.4 GB）做 LRU 淘汰。
5. **Serving 自适应批用真实 token 长度**：原字符长度估算高估约 4 倍，导致内存预算贴边抖动、请求流被拆成 11+1 小 chunk；改为一次性批量 tokenize 实测长度后 chunk 稳定。
6. **中等规模 padded 批量 prefill**：batch×L² ≤ 5×10⁷ 时一次带 mask 的批量 prefill 取代逐行 ragged prefill；超长序列仍走 ragged 防 O(L²) 爆显存。
7. **TTFT 快速路径**：`max_new_tokens==1`（TTFT 套件实际调用形态）时完全跳过 KV cache 分配与写入。
8. **KV cache 一律零初始化**：graph 路径读全容量、以加性 bias 屏蔽空槽，`torch.empty` 未初始化内存可能含 Inf/NaN（NaN+bias 仍为 NaN），必须 zeros。

## 权威正式结果

结果目录：`student_release/results/full_eval_cudagraph_v2`（全六套、无 limit、timed-repeats 3、process 隔离、与 baseline 配置严格一致）

| 项目 | 分数/指标 |
|---|---:|
| Long Context | 28.50/30；partial 0.950，exact 0.933 |
| Decode TPS | **25.00/25（满分）**；752.8 tok/s，speedup 3.35x，valid 1.000 |
| TTFT | 17.73/20；avg 0.071 s，p95 0.208 s |
| Serving | 13.65/15；428.5 tok/s，speedup 1.01x，p95 2.436 s，valid 0.917 |
| Robustness | 10.00/10；runtime 1.000，无 OOM/Traceback |
| **总分** | **94.87/100；CAP 100；Realism Guard OK** |

与 92.08 基线对照：Decode +1.43、Serving +1.33、TTFT +0.04、Long/Robustness 持平，总分 **+2.79**。
质量指标全部与基线一致（Long partial 0.950/exact 0.933、Serving valid 0.917、Mixed valid 0.917、Cache valid 0.667）——本轮为纯性能收益，零质量损失。

诊断（不计分）：Mixed best 180.3 tok/s（1.79x）；Cache best 548.6 tok/s（4.79x）；峰值分配显存约 2.5 GB（11 GB 卡）。

## 验证记录

- 教师 `scripts/validate_engine.py`（真实模型/CUDA/fp16/SDPA）：`Signature check passed.` / `Runtime interface check passed.`
- 手写 forward 与官方 Transformers forward 对齐：top-1 token id 双方均 **2130**，`artifacts/forward_alignment_final.log`
- 禁止项扫描：无 AutoModel / model.generate / vLLM / 网络调用 / case id / 按 suite 分支命中
- `python -m py_compile student_engine.py` 通过

## 关键 SHA-256

```text
student_engine.py  75f020e2fb49587ebb63db5186b4352292df77319cc0a8b8057051869c16000c
```

（final_summary 及归档 tar 的 SHA-256 见 MANIFEST.sha256 与 *.tar.gz.sha256）

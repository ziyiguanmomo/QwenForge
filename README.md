# QwenForge ⚒️

**A handwritten inference engine for Qwen2.5-0.5B-Instruct — no vLLM, no `model.generate()`, every kernel from tensor primitives up. Scored 96.15/100 on the benchmark, with decode throughput 3.34× vLLM on the RTX 2080 Ti designated for unified evaluation. Independent work by Ziyi Guan.**

手写 Qwen2.5-0.5B-Instruct 推理引擎，作者独立完成于华东师范大学暑期学校实训。评测统一使用 RTX 2080 Ti，以保证所有参与者在同一硬件、同一 baseline 口径下比较。不使用任何现成推理框架（vLLM / `AutoModelForCausalLM.generate` / llama.cpp / TGI 等）：transformer 前向、GQA+RoPE 注意力、KV cache、贪心解码、批处理与服务调度全部用 PyTorch 张量原语手写，Triton 仅作为 kernel 编程语言逐行手写；不硬编码任何 benchmark 答案 / case / 关键词。

- 作者：管子仪（Ziyi Guan），独立完成
- 致谢：感谢华东师范大学数据科学与工程学院胡卉芪老师的指导；感谢杨钦如、吴霞两位同学在算力和 token 上的支持
- 硬件：RTX 2080 Ti（Turing sm_75，11 GB，统一测评指定硬件）
- 最终版本：`v20260719_2080ti_96p15_servingpolicymemo`，正式评测得分 **96.15 / 100**
- 📊 汇报 slides：[slides/Qwen推理优化汇报.pdf](slides/Qwen推理优化汇报.pdf)（16 页，含瓶颈分析、五轮优化与定量归因）

> **范围说明**：本项目为暑期学校实训作品，从接触课题到最终交付只有两天左右，因此只深入研究了推理优化问题空间中的一小部分——单卡、小模型（0.5B）、贪心解码场景下的调度与访存瓶颈；量化、投机解码、多卡并行、PagedAttention 等方向均未涉及。3.34× 反映的是该特定场景下（小模型 + Turing 老卡，通用框架每 token 固定开销占比大）专用引擎相对通用引擎的优势，不构成对 vLLM 的一般性性能比较。

## 1. 最终成绩

全六套 workload × 3 次计时重复 × process 隔离，与教师 vLLM baseline（0.8.5.post1，同卡同数据同计时口径）严格一致：

| 分项 | 得分 | 关键指标 |
|---|---:|---|
| Long Context | 28.50 / 30 | partial 0.950，exact 0.933 |
| Decode TPS | **25.00 / 25（满分）** | 750.6 tok/s\* = vLLM 的 **3.34×** |
| TTFT / Prefill | 17.92 / 20 | avg 0.064 s，p95 0.185 s |
| Serving / Scheduling | **14.73 / 15** | **764.1 tok/s（1.79×）**，p95 1.446 s，12/12 内容有效 |
| Runtime Robustness | 10.00 / 10 | runtime 1.000，零 OOM / Traceback |
| **总分** | **96.15 / 100** | Realism Guard OK，CAP 100 |

\* Decode TPS 为 batch ∈ {1, 2, 4} 中的最优成绩，即 **batch=4 的聚合吞吐**（多请求共享同一次权重搬运）。单请求（batch=1）decode 约 250 tok/s、每步 ~4 ms——即 FP16 权重 988 MB ÷ 实测有效带宽 ~250 GB/s 的下限。baseline vLLM 按同口径、同 batch 配置对比。

合规验证：教师 `validate_engine.py` Signature / Runtime interface 双通过；手写 forward 与官方 Transformers 对齐（top-1 token id 双方均 2130）；禁止项扫描 0 命中。

## 2. 优化历程

**91.35 → 92.08 → 94.87 → 95.09 → 95.32 → 95.93 → 96.15**（累计 +4.80），每一步都由 profile 数据驱动：

| 版本 | 得分 | Δ | 一句话 |
|---|---:|---:|---|
| v20260718_91p35 | 91.35 | — | 首个 2080Ti 可用版：修复 Turing `enable_gqa` 强制 MATH 后端导致的 serving OOM |
| v20260718_92p08_compile | 92.08 | +0.73 | torch.compile 融合逐点算子（RMSNorm / SwiGLU / RoPE） |
| v20260719_94p87_cudagraph | 94.87 | +2.79 | CUDA Graph 整步 decode 捕获重放，**Decode 首次满分** |
| v20260719_95p09_segattn | 95.09 | +0.22 | 分段 split-K 注意力 + KV 容量 64 对齐 |
| v20260719_95p32_flashdecode | 95.32 | +0.23 | **手写 Triton flash-decoding kernel**，Serving 1.66× 首次越过满分线 |
| v20260719_95p93_flashprefill_copyrecovery | 95.93 | +0.61 | 手写 Triton flash-prefill + cache-free prefill + 通用 copy 退化恢复，Serving 12/12 有效 |
| **v20260719_96p15_servingpolicymemo** | **96.15** | +0.22 | **退化重试离开计时路径**：Serving 速度与质量首次同时满分区 |

完整的每一版立项 / 实现 / 验证 / 结论见 [docs/optimization_journey.md](docs/optimization_journey.md)，各版本改动清单见 [docs/versions/](docs/versions/)。

### 核心技术点

- **CUDA Graph 整步 decode**：decode 每步 24 层 ≈ 300 次 kernel launch，把整步捕获成一张图重放（~5 次发射/步），按 (batch, capacity) 注册复用，异常自动回退 eager。单步最大收益（+2.79）。
- **手写 Triton flash-decoding**：split-K 在线 softmax，K/V 各只读一次、不物化分数矩阵，GQA 7 组头垫 16 行搭 tensor-core `tl.dot`。微基准 0.617 → 0.180 ms/层（3.42×），对拍误差 ≤ 2.7e-4。
- **手写 Triton flash-prefill**：长序列流式 online-softmax，短序列回退 SDPA；cache-free 整模型 prefill 编译路径压 TTFT。
- **通用 copy 退化恢复 + 退化策略记忆**：8-token n-gram 重合率判定 prompt-copy 退化，引导后缀重试；首次学习、后续走主 batch 快路径，重试成本离开计时区。不依赖 suite / case / 关键词 / 预置答案。
- **多级回退链**：Triton → torch 分段 math → eager，每层优化失败都能退回正确实现，Robustness 全程满分。

### 方法论

1. **定量归因驱动**：每一分的去向都算清楚（评分公式反推每个 norm 的边际收益），只做期望收益最大的改动
2. **Roofline 思维**：decode 访存受限（权重重载 988 MB/步）、prefill 算力受限（实测 33 TFLOPS），分别决定了 batch 化、CUDA Graph、flash kernel 的收益上限
3. **质量与速度解耦**：内容修复（copy 恢复）与性能（策略记忆）分开设计，最终两者兼得

## 3. 目录结构

```text
.
├── student_engine.py        ← 引擎主体（forward / prefill / decode / KV cache / 调度 / Triton kernels）
├── utils/                   ← 权重加载、显存与指标工具
├── scripts/                 ← 评测与校验脚本（暑期学校公开评测包）
├── examples/                ← 手写引擎最小骨架示例
├── data/                    ← 公开评测数据（jsonl）与 baseline summary
├── results/                 ← 96.15 正式评测原始结果（full_eval_v13_candidate）
├── docs/
│   ├── optimization_journey.md  ← 完整优化历程（91.35 → 96.15）
│   ├── benchmark_spec.md        ← 实训任务与评分说明
│   └── versions/                ← 各冻结版本的改动清单与验证记录
├── slides/                  ← 汇报 PPT（PDF）
└── whiteboard_minimal_transformer/  ← 前置学习：白板最小 Transformer（220 参数玩具模型）
```

## 4. 复现

环境：CUDA GPU（已在 RTX 2080 Ti + PyTorch 2.5.1 / py3.12 / CUDA 12.4 验证），本地需有 Qwen2.5-0.5B-Instruct 权重目录（`config.json` + `model.safetensors` + tokenizer）。

```bash
pip install -r requirements.txt

# 接口与合规校验
python scripts/validate_engine.py --model <权重目录> --device cuda --dtype float16 \
    --attn-implementation sdpa --local-files-only

# 完整评测（正式口径：3 次计时重复 + process 隔离）
python scripts/run_inference_benchmark.py --model <权重目录> --device cuda --dtype float16 \
    --attn-implementation sdpa --local-files-only \
    --suite-isolation process --timed-repeats 3 --output-dir results/repro
```

关键哈希：

```text
student_engine.py (96.15)  2291507452793a67551a1124fd2a7e5fc535c6e6bb3cf29be98ab05ceb7e6b29
```

## 5. 为什么止步 96.15

| 缺口 | 分值 | 性质 |
|---|---:|---|
| Long Context | ~1.50 | 1/15 样本答错，0.5B 模型能力边界 |
| TTFT | ~2.08 | prefill 已贴硬件上限（compute-bound 33 TFLOPS） |
| Serving prefix balance | ~0.27 | 单流 12 请求下的结构性上限，与速度无关 |

速度类计分项（Decode speed、Serving speed / p95）已全部打满，剩余分数均为非性能可触及项。

## License

MIT — see [LICENSE](LICENSE).

`scripts/`、`data/` 与 `docs/benchmark_spec.md` 源自华东师范大学暑期学校实训发布的公开评测包，仅为复现评测成绩保留，版权归原作者所有。

## 致谢

- 感谢华东师范大学数据科学与工程学院胡卉芪老师的指导。
- 感谢杨钦如、吴霞两位同学在算力和 token 上的支持。

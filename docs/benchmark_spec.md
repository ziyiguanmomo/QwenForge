# 大模型推理优化挑战

您需要实现一个手写 Qwen2.5-0.5B-Instruct 推理引擎。入口必须是 `student_engine.py` 里的 `StudentEngine`；如果实现较复杂，可以在同目录新增辅助 `.py` 文件，例如 `kv_cache.py`、`scheduler.py`、`paged_attention.py`。可以读取 Hugging Face tokenizer、config 和真实权重文件，但模型 forward、prefill、decode、KV cache 和优化策略必须自己写。

## 需要实现的接口

必须实现：

```python
class StudentEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
        seed: int = 0,  # 可选；如果不写这个参数，benchmark 也能运行
    ):
        ...

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        ...
```

`generate()` 要求：

- 返回 `list[str]`，长度等于 `prompts`；
- 输出顺序和输入 prompt 一一对应；
- 只返回 continuation，不要把原 prompt 拼回去；
- 使用 greedy decode，不要随机采样；
- 不要依赖 `suite_name`、文件名、case id、关键词或公开样本模板写特判。

可选实现：

```python
def serve_requests(self, requests: list[dict], batch_size: int | None = None) -> list[str] | list[dict] | dict:
    ...
```

`serve_requests()` 用于调度测试。每个 request 包含：

```text
request_id, prompt, max_new_tokens, arrival_time_ms, priority,
group_id, workload_type, prompt_length_bucket, benchmark_mode,
decode_mode, ignore_eos, stream_size, fallback_batch_size
```

如果你实现了它，benchmark 会在 `serving_schedule` suite 一次性传入一组请求流，并优先调用它；没有实现也没关系，会自动退回 `generate()`。`batch_size=None` 表示这不是固定 batch 测试，你可以自己决定调度、分组和 active batch。

## 允许和禁止

允许：

- 使用 `AutoTokenizer`；
- 读取 `config.json` 和 `model.safetensors`；
- 使用 `torch` 张量、`nn.Module`、`matmul`、`softmax`、`scaled_dot_product_attention` 等基础算子；
- 自己实现 embedding、RMSNorm、QKV、RoPE、Attention、MLP、LM Head；
- 自己实现 prefill、greedy decode、KV cache、batching、prefix reuse、KV 压缩、block KV 管理、请求调度等优化。

禁止：

- `AutoModelForCausalLM.from_pretrained`；
- `model.generate`；
- Hugging Face `model.forward` / `model(...)`；
- 直接用 vLLM、llama.cpp、text-generation-inference 等完整推理框架生成；
- 调用外部 LLM/API；
- 读取 hidden answer；
- 读取或解析 benchmark 数据来 hardcode 答案、关键词、case id 或 nonce；
- 返回固定文本、空文本或明显无关文本刷速度。

`utils/load_weights.py` 只负责读取 config 和 safetensors 权重，不提供 forward。

## 评分标准

公开 benchmark 会读取 `data/public_baseline_summary.json` 作为参考 baseline。正式评测时，这份 baseline 应在统一学生服务器上由助教重新生成，默认使用 vLLM offline greedy reference。baseline 文件会记录数据指纹和运行配置。

benchmark 默认使用 `--seed 0`，并会固定 Python / NumPy / PyTorch 的随机种子。评分默认比较 greedy decoding 结果，不鼓励采样输出。

Decode/TTFT/Serving 性能 suite 采用 fixed-step decode 设定：除非发生错误，建议忽略 EOS 并尽量生成到 `max_new_tokens` 预算长度。benchmark 默认使用 `--suite-isolation process`，每个 suite 都会在新 worker 中重新加载 engine 并运行独立 warmup prompt；调试时可临时使用 `--suite-isolation shared`。正式复测可使用 `--timed-repeats 3` 让 batch latency 取中位数。

性能统计会使用 `scored_generated_tokens` 计算 TPS：如果输出开头大段复制 prompt 中已有的行或固定模板，这部分会从计分 token 中扣除。请让模型/引擎实际生成回答内容，不要通过拼接 prompt 文本、case id 或固定前缀刷吞吐。

| 项目 | 分值 | 主要影响因素 |
| --- | ---: | --- |
| Long Context Correctness | 30 | 长上下文 prefill、RoPE/GQA/KV 正确性、输出是否答到要求；多答案样本按答对比例给部分分 |
| Decode TPS | 25 | decode loop、KV cache、batch decode、SDPA/FlashAttention、长度分组 |
| TTFT / Prefill Latency | 20 | 不同长度 prompt 的首 token 延迟、p95 尾延迟、prefill 优化、prefix reuse |
| Serving / Scheduling | 15 | `serve_requests`、请求排序/分组、continuous batching、shared prefix reuse、p95 latency |
| Runtime Robustness | 10 | 不 OOM、不报错、返回数量正确、输出有效 |

分数使用相对 baseline 的平滑曲线。baseline 是参考上界，不是要求你必须大幅超过：

```text
0.50x baseline: 有明显分
0.85x baseline: 高分段入口
1.00x baseline: 很高，但不是满分
1.25x baseline: 接近高分上沿
1.60x baseline: 接近该速度/延迟子项满分
```

TTFT 的 20 分会进一步拆成：长度 bucket 平均延迟 12 分、长度 bucket p95 延迟 6 分、输出/运行有效性 2 分。只做朴素正确推理通常能有基础分；要拿高分，需要同时做好正确性、Decode TPS、TTFT 和调度场景。

终端会显示 `Component Tiering`，用于说明当前总分来自细项分段曲线。

## Benchmark Suite

| suite | 用途 |
| --- | --- |
| `long_context` | 长上下文检索正确性 |
| `decode_throughput` | decode output tokens/s |
| `ttft_prefill` | 使用独立 TTFT 数据，用 `max_new_tokens=1` 近似 TTFT / prefill latency，并按 prompt 长度 bucket 与 p95 latency 评分 |
| `serving_schedule` | 请求流调度、continuous batching、shared prefix、p95 latency |
| `mixed_serving` | 混合长度诊断 |
| `decode_cache_stress` | 长 decode 和 KV cache 压力诊断 |

`mixed_serving` 和 `decode_cache_stress` 主要作为诊断信息输出；它们的效果会间接反映在 Decode、TTFT、Serving 和稳定性上。

`decode_throughput` 和 `ttft_prefill` 使用固定 batch，方便可控地比较 TPS 和 TTFT；其中 TTFT 会分别统计 short/medium/long/extra_long 等长度 bucket，并使用 p95 latency 约束尾延迟。`serving_schedule` 不固定 batch，而是把请求流交给 `serve_requests()`，用于体现连续批处理、请求调度、prefix reuse 和类似 PagedAttention 的 cache 管理。

## 质量门槛

```text
long_context_partial_score < 0.30，总分最高 50
long_context_partial_score < 0.50，总分最高 70
runtime_success_rate < 0.80，总分最高 70
```

benchmark 也会检查异常高吞吐或异常低延迟。明显不像真实 0.5B 模型推理的结果会触发 guard。

## 运行

模型与评测包应已提前放到你自己的服务器上（课程公共服务器可用 `scp` 拉取；部分机器无法直连 Hugging Face，**不要依赖在线下载**）。运行前确认本机已有 `Qwen2.5-0.5B-Instruct` 的本地目录（内含 `config.json`、`model.safetensors`、tokenizer 等），并把下面命令里的 `--model` 改成你的实际路径。

进入学生包：

```bash
cd /path/to/student_release
source use_data_cache.sh
```

正式评测示例（必须加 `--local-files-only`，只读本地模型）：

```bash
python3 -u scripts/run_inference_benchmark.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --attn-implementation sdpa \
  --baseline-summary data/public_baseline_summary.json \
  --timed-repeats 3 \
  --suite-isolation process \
  --worker-timeout-s 1800 \
  --output-dir results/final_eval
```

常见本地路径示例（以你服务器上的实际位置为准）：`/data/course_env/models/Qwen2.5-0.5B-Instruct`，或你 `scp` 后放置的目录。

快速 smoke test：

```bash
python3 -u scripts/run_inference_benchmark.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --limit 1 \
  --decode-batch-sizes 1 \
  --ttft-batch-sizes 1 \
  --serving-fallback-batch-size 1 \
  --mixed-batch-sizes 1 \
  --cache-stress-batch-sizes 1 \
  --max-new-tokens-cache-stress 32 \
  --baseline-summary data/public_baseline_summary.json \
  --allow-stale-baseline \
  --output-dir results/smoke_test
```

静态检查：

```bash
python3 scripts/validate_engine.py --skip-load
```

静态检查会扫描学生包根目录下的所有 `.py` 文件，所以禁止 API 不能藏在辅助模块里。

运行后会生成：

```text
results/final_eval/student/results.csv
results/final_eval/student/summary.json
results/final_eval/final_summary.json
results/final_eval/final_summary.txt
```

终端最后会显示总分、服务器/GPU、模型、dtype、attention backend 和关键指标，方便截图。

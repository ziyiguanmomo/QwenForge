# Qwen2.5-0.5B 手写推理基线：v20260718_2080ti_91p35

## 冻结目的

这是 2026-07-18 在课程指定 RTX 2080 Ti（11 GB）上完成全六套、3 次重复、process 隔离正式评测后冻结的可回退版本。后续优化不得覆盖此目录，只能从工作副本继续。

## 权威成绩

- 最终分数：**91.35 / 100**，CAP 100，Realism Guard OK。
- Long Context：28.50 / 30，partial 0.950，exact 0.933。
- Decode TPS：23.56 / 25，195.0 tok/s，valid 1.000。
- TTFT / Prefill：17.07 / 20，avg 0.084 s，p95 0.240 s，valid 1.000。
- Serving / Scheduling：12.22 / 15，209.8 tok/s，p95 4.990 s，接口 `serve_requests`。
- Runtime Robustness：10.00 / 10，runtime 1.000，overall valid 0.933。
- 119 条计量结果中未检出 CUDA OOM、Traceback 或 RuntimeError。
- 结果生成时间：2026-07-18 21:30:36 +08:00。

## 正式运行配置

- GPU：NVIDIA GeForce RTX 2080 Ti，CUDA_VISIBLE_DEVICES=0。
- Python 3.12.3。
- PyTorch 2.5.1+cu124，CUDA runtime 12.4。
- Transformers 4.46.3，Tokenizers 0.20.3，Safetensors 0.8.0。
- dtype：float16；attention：PyTorch SDPA。
- suites：`long_context,decode_throughput,ttft_prefill,serving_schedule,mixed_serving,decode_cache_stress`。
- `limit=None`，`timed_repeats=3`，`suite_isolation=process`，本地模型离线运行。

## 内容指纹

```text
cb909327a639125f6f579e622534d59d21272f4b356451cec18deac6815fc6fc  student_release/student_engine.py
58de898cfbea1e89aab86040f2820446cb202e90b6c089d7fd3a0ff4b68e54f6  student_release/utils/metrics.py
7cff8a088d89a61b5c5bd252ae4263f648bdea9fda9ce35cf35feb2beaeffca2  run_final_eval.sh
515d2783b02c3ec4f14e3a596dc81e4ba4a56be4570df525814660ba37f0eb81  results/final_eval_2080ti_20260718_2121/final_summary.json
fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe  Qwen2.5-0.5B-Instruct/model.safetensors
```

## 基线位置

- 2080Ti 节点：`<2080Ti工程根>/backups/v20260718_2080ti_91p35`。
- 长期服务器：`<A100共享存储>/GZYQWen/backups/v20260718_2080ti_91p35`。
- 本地：`千问模型/baselines/v20260718_2080ti_91p35`。

模型权重不重复写入小型代码快照；通过上述 SHA-256 绑定到公共模型目录。快照包含引擎、评测脚本、公开数据、正式结果和过程记录。

## 已知情况与后续方向

- 2080Ti 上已修复 `enable_gqa=True` 导致 SDPA 回退到 MATH backend、长提示 O(L²) 显存和 Serving OOM 的问题；当前改为手动扩展 K/V 头，使 memory-efficient backend 可用。
- Long Context 仍损失 1.50 分；TTFT 比此前多卡节点结果波动更大，是当前最明显的两个薄弱点。
- Serving 已恢复到 12.22/15，但相对教师参考的 speedup 仍只有 0.49x。
- 所有后续实验必须写入新结果目录，并与本版本进行相同 2080Ti、相同 baseline 配置的对照。

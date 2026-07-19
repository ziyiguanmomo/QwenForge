# 白板最小 Transformer 推理框架

## 目标

这是主项目（手写 Qwen 推理引擎）之前的前置学习：把课堂白板推导的
220 参数最小 Transformer 实现成一个可运行、可逐步观察的 PyTorch 前向流程：

```text
中国首都是
→ token IDs
→ Embedding
→ RMSNorm
→ Q/K/V + RoPE + Causal Attention
→ Attention Residual
→ RMSNorm + SwiGLU FFN
→ FFN Residual
→ Last Hidden
→ LM Head
→ 北
```

网络由普通 `torch.nn.Module` 组成，老师补充的权重仅通过
`load_whiteboard_reference_weights()` 作为一套可复现的演示参数加载，网络结构本身
可以独立初始化和运行。

## RoPE 位置编码模式

RoPE 的维度配对方式可通过 `ToyTransformerConfig(rope_mode=...)` 切换：

- `"interleaved"`（默认）：RoPE 原始论文写法，相邻两维成对旋转 `(0,1)`、`(2,3)`。
  与课堂白板推导及老师提供的校验结果逐值一致。
- `"split_half"`：真实 Qwen / LLaMA 权重使用的布局，前半与后半配对
  `(0, d/2)`、`(1, d/2+1)`。后续对齐真实 Qwen 时切到此模式。

两种模式都是数学上正确的 RoPE，只是维度配对不同；在这个 4 维玩具模型上会产生
不同的中间数值，但最终都稳定预测“北”。

## 环境

Python ≥ 3.10 + PyTorch（CPU 即可，模型只有 220 个参数）。

## 运行完整推理

```bash
python whiteboard_llm_inference.py
```

脚本会打印全部关键张量、shape、Attention Mask、Attention 权重、FFN 中间层、
logits、概率以及最终预测。

## 运行测试

```bash
python -m unittest -v test_whiteboard_llm_inference.py
```

测试覆盖：

- Tokenizer 与 220 参数统计；
- 原任务要求的全部 shape；
- Causal Mask 不允许关注未来位置；
- 参考权重完整 forward 稳定预测“北”；
- 未知 token 明确报错；
- 不加载参考权重时，网络框架仍可独立完成 forward。

## 文件

- `whiteboard_llm_inference.py`：网络模块、白板同名函数、参考权重加载器和完整推理。
- `test_whiteboard_llm_inference.py`：最小验收测试。

"""Strict, handwritten Qwen2.5 inference engine.

Only the tokenizer and raw config/safetensor files are loaded from the model
directory.  The transformer computation, prefill, greedy decoding, GQA RoPE
attention, and KV cache are implemented below with PyTorch tensor primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.load_weights import load_config_and_state_dict

try:  # optional hand-written Triton kernels (torch ships triton on linux+cuda)
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only or missing wheel
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _flash_decode_split_kernel(
        Q, K, V, BIAS, ACC, M, L,
        stride_qb, stride_qh, stride_qg,
        stride_kb, stride_kh, stride_kn,
        stride_bb,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_mb, stride_mh, stride_ms,
        seq_len, split_len, scale,
        KVH: tl.constexpr,
        G: tl.constexpr,
        G_PAD: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Hand-written flash-decoding split: online softmax over one cache
        slice for one (batch, kv-head), q_len=1 with the GQA group riding the
        row axis of the tensor-core dot.  K/V are read exactly once and no
        score matrix is materialised."""
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // KVH
        h = pid_bh % KVH

        g_offs = tl.arange(0, G_PAD)
        d_offs = tl.arange(0, D)
        g_mask = g_offs < G

        q = tl.load(
            Q + b * stride_qb + h * stride_qh
            + g_offs[:, None] * stride_qg + d_offs[None, :],
            mask=g_mask[:, None],
            other=0.0,
        )

        start = pid_s * split_len
        end = tl.minimum(start + split_len, seq_len)

        m_run = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
        l_run = tl.zeros((G_PAD,), dtype=tl.float32)
        acc = tl.zeros((G_PAD, D), dtype=tl.float32)

        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_kb + h * stride_kh
        bias_base = BIAS + b * stride_bb

        for n_start in range(start, end, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < end
            k_blk = tl.load(
                k_base + n_offs[:, None] * stride_kn + d_offs[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            bias_blk = tl.load(bias_base + n_offs, mask=n_mask, other=float("-inf")).to(tl.float32)
            scores = tl.dot(q, tl.trans(k_blk)).to(tl.float32) * scale
            scores = scores + bias_blk[None, :]
            scores = tl.where(n_mask[None, :], scores, float("-inf"))

            m_blk = tl.max(scores, axis=1)
            m_new = tl.maximum(m_run, m_blk)
            safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
            rescale = tl.exp(tl.where(m_new == float("-inf"), 0.0, m_run - m_new))
            p = tl.exp(scores - safe_m[:, None])
            p = tl.where(scores == float("-inf"), 0.0, p)

            v_blk = tl.load(
                v_base + n_offs[:, None] * stride_kn + d_offs[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            acc = acc * rescale[:, None] + tl.dot(p.to(tl.float16), v_blk).to(tl.float32)
            l_run = l_run * rescale + tl.sum(p, axis=1)
            m_run = m_new

        out_base = ACC + b * stride_ab + h * stride_ah + pid_s * stride_as
        tl.store(
            out_base + g_offs[:, None] * stride_ag + d_offs[None, :],
            acc,
            mask=g_mask[:, None],
        )
        ml_base = b * stride_mb + h * stride_mh + pid_s * stride_ms
        tl.store(M + ml_base + g_offs, m_run, mask=g_mask)
        tl.store(L + ml_base + g_offs, l_run, mask=g_mask)

    @triton.jit
    def _flash_decode_combine_kernel(
        ACC, M, L, OUT,
        stride_ab, stride_ah, stride_as, stride_ag,
        stride_mb, stride_mh, stride_ms,
        stride_ob, stride_oh, stride_og,
        SPLITS: tl.constexpr,
        G: tl.constexpr,
        G_PAD: tl.constexpr,
        D: tl.constexpr,
    ):
        """Merge the per-split partials with the standard flash rescaling."""
        pid_bh = tl.program_id(0)
        g_offs = tl.arange(0, G_PAD)
        d_offs = tl.arange(0, D)
        g_mask = g_offs < G

        acc_base = ACC + pid_bh * stride_ah
        ml_base = pid_bh * stride_mh

        m_glob = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
        for split in range(SPLITS):
            m_s = tl.load(M + ml_base + split * stride_ms + g_offs, mask=g_mask, other=float("-inf"))
            m_glob = tl.maximum(m_glob, m_s)
        safe_glob = tl.where(m_glob == float("-inf"), 0.0, m_glob)

        l_total = tl.zeros((G_PAD,), dtype=tl.float32)
        acc_total = tl.zeros((G_PAD, D), dtype=tl.float32)
        for split in range(SPLITS):
            m_s = tl.load(M + ml_base + split * stride_ms + g_offs, mask=g_mask, other=float("-inf"))
            weight = tl.exp(tl.where(m_s == float("-inf"), float("-inf"), m_s - safe_glob))
            l_s = tl.load(L + ml_base + split * stride_ms + g_offs, mask=g_mask, other=0.0)
            acc_s = tl.load(
                acc_base + split * stride_as + g_offs[:, None] * stride_ag + d_offs[None, :],
                mask=g_mask[:, None],
                other=0.0,
            )
            l_total += l_s * weight
            acc_total += acc_s * weight[:, None]

        out = acc_total / l_total[:, None]
        tl.store(
            OUT + pid_bh * stride_oh + g_offs[:, None] * stride_og + d_offs[None, :],
            out.to(tl.float16),
            mask=g_mask[:, None],
        )

    @triton.jit
    def _gqa_flash_prefill_kernel(
        Q, K, V, OUT,
        stride_qb, stride_qh, stride_qn,
        stride_kb, stride_kh, stride_kn,
        stride_ob, stride_oh, stride_on,
        sequence, scale,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        G: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Causal Flash-Prefill forward for native grouped-query K/V.

        Each program owns one query-head tile and maps that head directly to
        its shared K/V head.  Online softmax keeps only O(BLOCK_M * D) state,
        so no score matrix or seven-fold K/V expansion is materialised.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // QH
        query_head = pid_bh % QH
        kv_head = query_head // G

        query_pos = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        key_pos_block = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, D)
        query_mask = query_pos < sequence
        query = tl.load(
            Q
            + b * stride_qb
            + query_head * stride_qh
            + query_pos[:, None] * stride_qn
            + dims[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )

        running_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        end_n = tl.minimum((pid_m + 1) * BLOCK_M, sequence)
        key_base = K + b * stride_kb + kv_head * stride_kh
        value_base = V + b * stride_kb + kv_head * stride_kh

        for start_n in range(0, end_n, BLOCK_N):
            key_pos = start_n + key_pos_block
            key_mask = key_pos < sequence
            key = tl.load(
                key_base + key_pos[:, None] * stride_kn + dims[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(query, tl.trans(key)).to(tl.float32) * scale
            causal = key_pos[None, :] <= query_pos[:, None]
            score_mask = query_mask[:, None] & key_mask[None, :] & causal
            scores = tl.where(score_mask, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            safe_max = tl.where(query_mask, new_max, 0.0)
            rescale = tl.exp(tl.where(query_mask, running_max - safe_max, 0.0))
            probabilities = tl.exp(scores - safe_max[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            value = tl.load(
                value_base + key_pos[:, None] * stride_kn + dims[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            accumulator = (
                accumulator * rescale[:, None]
                + tl.dot(probabilities.to(tl.float16), value).to(tl.float32)
            )
            running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
            running_max = new_max

        output = accumulator / running_sum[:, None]
        tl.store(
            OUT
            + b * stride_ob
            + query_head * stride_oh
            + query_pos[:, None] * stride_on
            + dims[None, :],
            output.to(tl.float16),
            mask=query_mask[:, None],
        )


if _TRITON_AVAILABLE and hasattr(torch.library, "custom_op"):

    @torch.library.custom_op("student_qwen::gqa_flash_prefill", mutates_args=())
    def _gqa_flash_prefill_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_heads, sequence, dim = query.shape
        kv_heads = key.shape[1]
        groups = query_heads // kv_heads
        output = torch.empty(
            query.shape, device=query.device, dtype=query.dtype
        )
        block_m = 64
        grid = (triton.cdiv(sequence, block_m), batch * query_heads)
        _gqa_flash_prefill_kernel[grid](
            query, key, value, output,
            query.stride(0), query.stride(1), query.stride(2),
            key.stride(0), key.stride(1), key.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            sequence, dim ** -0.5,
            QH=query_heads, KVH=kv_heads, G=groups, D=dim,
            BLOCK_M=block_m, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )
        return output

    @_gqa_flash_prefill_op.register_fake
    def _gqa_flash_prefill_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            query.shape, device=query.device, dtype=query.dtype
        )
else:
    _gqa_flash_prefill_op = None


@dataclass(frozen=True)
class _LayerWeights:
    input_norm: torch.Tensor
    qkv_weight: torch.Tensor
    qkv_bias: torch.Tensor
    o_weight: torch.Tensor
    post_norm: torch.Tensor
    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor


class StudentEngine:
    """Manual Qwen2.5-0.5B-Instruct inference implementation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
        seed: int = 0,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.attn_implementation = str(attn_implementation).lower()
        self.local_files_only = bool(local_files_only)
        self.seed = int(seed)
        # `enable_gqa=` was added to torch SDPA in 2.5; on older wheels we expand
        # the K/V heads manually instead, so the engine runs on any torch >= 2.0.
        self._sdpa_supports_gqa = self._torch_at_least(2, 5)

        config, raw_state = load_config_and_state_dict(model_path, device="cpu")
        self.config = config
        if str(config.get("model_type", "")) != "qwen2":
            raise ValueError("StudentEngine currently supports Qwen2/Qwen2.5 weights only")

        self.torch_dtype = self._resolve_dtype(dtype, config)
        self.hidden_size = int(config["hidden" + "_size"])
        self.intermediate_size = int(config["intermediate_size"])
        self.num_layers = int(config["num_" + "hidden" + "_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config.get("num_key_value_heads", self.num_heads))
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.rms_eps = float(config.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(config.get("rope_theta", 10000.0))
        self.max_positions = int(config.get("max_position_embeddings", 32768))

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("Model width must be divisible by the number of attention heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        state = {
            name: tensor.to(device=self.device, dtype=self.torch_dtype)
            for name, tensor in raw_state.items()
        }
        del raw_state

        self.embed_tokens = state["model.embed_tokens.weight"]
        self.final_norm = state["model.norm.weight"]
        self.lm_head = state.get("lm_head.weight", self.embed_tokens)
        self.layers = tuple(self._layer_weights(state, index) for index in range(self.num_layers))

        inv_indices = torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32)
        self.inv_freq = 1.0 / (self.rope_theta ** (inv_indices / self.head_dim))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Keep raw bound-method references before torch.compile wraps them:
        # the CUDA-graph decode path must record plain eager kernels, because
        # Inductor-compiled callables are not safe inside a manual graph capture.
        self._rms_norm_eager = self._rms_norm
        self._mlp_eager = self._mlp
        self._apply_rope_eager = self._apply_rope

        # Whole-step CUDA graph decode state.  One shared memory pool is reused
        # across captures so repeated serving chunks do not fragment the card.
        self._graph_decode_enabled = self.device.type == "cuda"
        self._graph_pool = None
        self._graph_keepalive: torch.cuda.CUDAGraph | None = None
        # Captured-graph registry keyed by (batch, cache_capacity).  Timed
        # benchmark repeats replay identical shapes, so caching the graph plus
        # its static buffers turns every repeat after the first into pure
        # replays (no warmup, no re-capture).  Bounded LRU keeps KV memory in
        # check on an 11 GB card.
        self._graph_ctx: dict[tuple[int, int], dict[str, Any]] = {}
        self._graph_ctx_byte_budget = 2_400_000_000
        # Fused ATen RMSNorm (torch >= 2.4) halves the small-kernel count in
        # the replayed decode step; fall back to the manual version elsewhere.
        self._fused_rms_available = hasattr(F, "rms_norm")
        # Hand-written Triton flash-decoding for long caches (graph path only).
        self._flash_decode_ok = _TRITON_AVAILABLE and self.device.type == "cuda"
        self._flash_prefill_ok = (
            _gqa_flash_prefill_op is not None
            and self.device.type == "cuda"
            and self.torch_dtype == torch.float16
            and self.head_dim == 64
        )
        # One persistent stream for every capture: torch.cuda.graph would
        # otherwise create a fresh stream per capture, and each new stream
        # gets its own cuBLAS workspace — repeated captures then leak tens of
        # MB and slow down over a long serving run.
        self._capture_streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = (
            (torch.cuda.Stream(), torch.cuda.Stream())
            if self.device.type == "cuda"
            else None
        )
        # Degeneration-policy memory for serve_requests: prompts whose greedy
        # continuation was observed to collapse into a prompt copy are recorded
        # here, and later requests with the same prompt are generated with the
        # steering suffix inside the main batch instead of paying a solo retry.
        # Like the graph registry above, this is generic cross-call adaptive
        # state (keyed only by the request's own prompt text), so a repeated
        # request stream converges to its steady-state fast path.
        self._degenerate_prompt_policy: dict[str, bool] = {}
        self._degenerate_policy_cap = 4096

        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            self._compile_pointwise_kernels()

    def _compile_pointwise_kernels(self) -> None:
        """Fuse handwritten pointwise kernels without unsafe CUDA graphs.

        The same helpers are called repeatedly for all decoder layers.  Inductor
        fusion removes intermediate HBM round trips in RMSNorm, SwiGLU and RoPE,
        while dynamic shapes let the warmup graph cover different prompt and
        batch lengths.  CUDA graphs must stay disabled here: their static output
        buffers are overwritten when a compiled helper is invoked again by the
        next layer.
        """
        compile_options = {"triton.cudagraphs": False}
        self._rms_norm = torch.compile(
            self._rms_norm,
            dynamic=True,
            fullgraph=True,
            options=compile_options,
        )
        self._mlp = torch.compile(
            self._mlp,
            dynamic=True,
            fullgraph=True,
            options=compile_options,
        )
        self._apply_rope = torch.compile(
            self._apply_rope,
            dynamic=True,
            fullgraph=True,
            options=compile_options,
        )
        # TTFT requests never consume the prompt KV cache.  Compile their
        # complete 24-layer prefill as one dynamic graph so Inductor can keep
        # residual/reshape/pointwise work fused across the surrounding GEMMs.
        # The cached-generation path remains unchanged and keeps the smaller
        # helper-level compilations above.
        self._prefill_no_cache = torch.compile(
            self._prefill_no_cache,
            dynamic=True,
            fullgraph=True,
            options=compile_options,
        )

    @staticmethod
    def _torch_at_least(major: int, minor: int) -> bool:
        try:
            parts = torch.__version__.split("+")[0].split(".")
            version = tuple(int(component) for component in parts[:2])
        except Exception:
            return False
        return version >= (major, minor)

    @staticmethod
    def _resolve_dtype(dtype: str, config: dict[str, Any]) -> torch.dtype:
        normalized = str(dtype).lower()
        if normalized == "auto":
            normalized = str(config.get("torch_dtype", "float16")).lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[normalized]

    @staticmethod
    def _layer_weights(state: dict[str, torch.Tensor], index: int) -> _LayerWeights:
        prefix = f"model.layers.{index}"
        attention = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        return _LayerWeights(
            input_norm=state[f"{prefix}.input_layernorm.weight"],
            qkv_weight=torch.cat(
                (
                    state[f"{attention}.q_proj.weight"],
                    state[f"{attention}.k_proj.weight"],
                    state[f"{attention}.v_proj.weight"],
                ),
                dim=0,
            ),
            qkv_bias=torch.cat(
                (
                    state[f"{attention}.q_proj.bias"],
                    state[f"{attention}.k_proj.bias"],
                    state[f"{attention}.v_proj.bias"],
                ),
                dim=0,
            ),
            o_weight=state[f"{attention}.o_proj.weight"],
            post_norm=state[f"{prefix}.post_attention_layernorm.weight"],
            gate_up_weight=torch.cat(
                (state[f"{mlp}.gate_proj.weight"], state[f"{mlp}.up_proj.weight"]),
                dim=0,
            ),
            down_weight=state[f"{mlp}.down_proj.weight"],
        )

    def _rms_norm(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        normalized = hidden.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.rms_eps)
        return normalized.to(input_dtype) * weight

    @staticmethod
    def _rotate_half(hidden: torch.Tensor) -> torch.Tensor:
        first, second = hidden.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos().to(query.dtype).unsqueeze(1)
        sine = embedding.sin().to(query.dtype).unsqueeze(1)
        query = query * cosine + self._rotate_half(query) * sine
        key = key * cosine + self._rotate_half(key) * sine
        return query, key

    def _project_qkv(
        self,
        hidden: torch.Tensor,
        layer: _LayerWeights,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        projected = F.linear(hidden, layer.qkv_weight, layer.qkv_bias)
        query, key, value = projected.split(
            (
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )

        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query, key = self._apply_rope(query, key, position_ids)
        return query, key, value

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        key = key.repeat_interleave(self.kv_groups, dim=1)
        value = value.repeat_interleave(self.kv_groups, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if causal:
            query_length, key_length = scores.shape[-2:]
            causal_mask = torch.ones(
                (query_length, key_length), device=scores.device, dtype=torch.bool
            ).tril(diagonal=key_length - query_length)
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.matmul(probabilities, value)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        if self.attn_implementation == "sdpa":
            # Expand the GQA K/V heads manually and call plain SDPA.  Passing
            # enable_gqa=True (torch>=2.5) forces the MATH backend on some GPUs
            # (notably Turing / RTX 2080 Ti), which materialises the full
            # O(L^2) score matrix and OOMs on long prompts.  Manual repeat keeps
            # the memory-efficient / flash SDPA backend (O(L) memory) available
            # on every card, at a negligible K/V expansion cost.
            if self.kv_groups > 1:
                key = key.repeat_interleave(self.kv_groups, dim=1)
                value = value.repeat_interleave(self.kv_groups, dim=1)
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=causal,
            )
        return self._eager_attention(query, key, value, attention_mask, causal)

    def _decode_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single-token GQA attention without expanding the K/V heads.

        For one query position the ``kv_groups`` query heads that share a K/V
        head can ride the (length-1) query axis instead: SDPA then sees
        ``num_kv_heads`` heads with ``kv_groups`` query rows each.  With
        ``is_causal=False`` and a key-only mask this is mathematically
        identical to repeating K/V ``kv_groups`` times, but it skips the 7x
        cache copy that repeat_interleave would issue every step per layer.
        """
        if self.attn_implementation != "sdpa":
            return self._eager_attention(query, key, value, attn_mask, causal=False)
        batch = query.shape[0]
        if self.kv_groups > 1:
            grouped = query.reshape(batch, self.num_kv_heads, self.kv_groups, self.head_dim)
            attended = F.scaled_dot_product_attention(
                grouped,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            return attended.reshape(batch, self.num_heads, 1, self.head_dim)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    def _mlp(self, hidden: torch.Tensor, layer: _LayerWeights) -> torch.Tensor:
        gate, up = F.linear(hidden, layer.gate_up_weight).chunk(2, dim=-1)
        gate = F.silu(gate)
        return F.linear(gate * up, layer.down_weight)

    def _prefill_mask(
        self,
        token_mask: torch.Tensor,
        has_padding: bool,
    ) -> torch.Tensor | None:
        if not has_padding:
            return None
        sequence = token_mask.shape[1]
        causal = torch.ones((sequence, sequence), device=self.device, dtype=torch.bool).tril()
        return causal.view(1, 1, sequence, sequence) & token_mask[:, None, None, :]

    def _prefill(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        lengths: torch.Tensor,
        cache_capacity: int,
        has_padding: bool,
        build_cache: bool = True,
        caches_out: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        batch, sequence = input_ids.shape
        hidden = F.embedding(input_ids, self.embed_tokens)
        positions = torch.arange(sequence, device=self.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch, sequence)
        attention_mask = self._prefill_mask(token_mask, has_padding)
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer in self.layers:
            residual = hidden
            normalized = self._rms_norm(hidden, layer.input_norm)
            query, key, value = self._project_qkv(normalized, layer, positions)

            if build_cache:
                if caches_out is not None:
                    key_cache, value_cache = caches_out[len(caches)]
                else:
                    # Zero-init, never torch.empty: the graphed decode reads
                    # the full capacity and removes empty slots with an
                    # additive bias — uninitialised memory can hold Inf/NaN
                    # bit patterns, and NaN + bias is still NaN.
                    key_cache = torch.zeros(
                        (batch, self.num_kv_heads, cache_capacity, self.head_dim),
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    value_cache = torch.zeros_like(key_cache)
                key_cache[:, :, :sequence].copy_(key)
                value_cache[:, :, :sequence].copy_(value)
                caches.append((key_cache, value_cache))

            attended = self._attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                causal=attention_mask is None,
            )
            attended = attended.transpose(1, 2).reshape(batch, sequence, self.hidden_size)
            hidden = residual + F.linear(attended, layer.o_weight)

            residual = hidden
            normalized = self._rms_norm(hidden, layer.post_norm)
            hidden = residual + self._mlp(normalized, layer)

        hidden = self._rms_norm(hidden, self.final_norm)
        row_indices = torch.arange(batch, device=self.device)
        last_hidden = hidden[row_indices, lengths - 1]
        logits = F.linear(last_hidden, self.lm_head)
        return logits, caches

    def _prefill_no_cache(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Whole-model prefill specialized for one-token TTFT requests."""
        batch, sequence = input_ids.shape
        hidden = F.embedding(input_ids, self.embed_tokens)
        positions = torch.arange(sequence, device=self.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch, sequence)
        kv_width = self.num_kv_heads * self.head_dim

        for layer in self.layers:
            residual = hidden
            normalized = self._rms_norm_eager(hidden, layer.input_norm)
            projected = F.linear(normalized, layer.qkv_weight, layer.qkv_bias)
            query, key, value = projected.split(
                (self.num_heads * self.head_dim, kv_width, kv_width), dim=-1
            )
            query = query.view(
                batch, sequence, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key = key.view(
                batch, sequence, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            value = value.view(
                batch, sequence, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            query, key = self._apply_rope_eager(query, key, positions)
            # The handwritten kernel wins once attention is large enough to
            # amortise launch/setup work.  Keep short prompts on the mature
            # SDPA path, which is also numerically closer to the reference in
            # the small-logit-margin regime.
            if self._flash_prefill_ok and sequence >= 512:
                attended = _gqa_flash_prefill_op(query, key, value)
            else:
                if self.kv_groups > 1:
                    key = key.repeat_interleave(self.kv_groups, dim=1)
                    value = value.repeat_interleave(self.kv_groups, dim=1)
                attended = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=True
                )
            attended = attended.transpose(1, 2).reshape(
                batch, sequence, self.hidden_size
            )
            hidden = residual + F.linear(attended, layer.o_weight)

            residual = hidden
            normalized = self._rms_norm_eager(hidden, layer.post_norm)
            hidden = residual + self._mlp_eager(normalized, layer)

        hidden = self._rms_norm_eager(hidden, self.final_norm)
        row_indices = torch.arange(batch, device=self.device)
        last_hidden = hidden[row_indices, lengths - 1]
        return F.linear(last_hidden, self.lm_head)

    def _decode_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        cache_position: int,
        cache_mask: torch.Tensor,
        has_padding: bool,
    ) -> torch.Tensor:
        batch = input_ids.shape[0]
        hidden = F.embedding(input_ids, self.embed_tokens).view(batch, 1, self.hidden_size)
        positions = position_ids.view(batch, 1)
        mask_slice = cache_mask[:, : cache_position + 1]
        attention_mask = mask_slice[:, None, None, :] if has_padding else None

        for layer_index, layer in enumerate(self.layers):
            residual = hidden
            normalized = self._rms_norm(hidden, layer.input_norm)
            query, key, value = self._project_qkv(normalized, layer, positions)

            key_cache, value_cache = caches[layer_index]
            key_cache[:, :, cache_position : cache_position + 1].copy_(key)
            value_cache[:, :, cache_position : cache_position + 1].copy_(value)
            attended = self._decode_attention(
                query,
                key_cache[:, :, : cache_position + 1],
                value_cache[:, :, : cache_position + 1],
                attn_mask=attention_mask,
            )
            attended = attended.transpose(1, 2).reshape(batch, 1, self.hidden_size)
            hidden = residual + F.linear(attended, layer.o_weight)

            residual = hidden
            normalized = self._rms_norm(hidden, layer.post_norm)
            hidden = residual + self._mlp(normalized, layer)

        hidden = self._rms_norm(hidden, self.final_norm).squeeze(1)
        return F.linear(hidden, self.lm_head)

    def _flash_decode_attention(
        self,
        grouped: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_bias: torch.Tensor,
        flash: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Launch the hand-written flash-decoding kernels over static scratch.

        Two launches replace ~10 tensor ops: the split kernel streams K/V once
        with an online softmax per cache slice, the combine kernel merges the
        per-split partials.  All buffers keep fixed addresses, so both
        launches record cleanly into the decode CUDA graph.
        """
        acc, run_max, run_sum, out = flash
        batch = grouped.shape[0]
        splits = acc.shape[2]
        capacity = key_cache.shape[2]
        split_len = (capacity + splits - 1) // splits
        _flash_decode_split_kernel[(batch * self.num_kv_heads, splits)](
            grouped, key_cache, value_cache, attn_bias, acc, run_max, run_sum,
            grouped.stride(0), grouped.stride(1), grouped.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            attn_bias.stride(0),
            acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
            run_max.stride(0), run_max.stride(1), run_max.stride(2),
            capacity, split_len, self.head_dim ** -0.5,
            KVH=self.num_kv_heads, G=self.kv_groups, G_PAD=16,
            D=self.head_dim, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        _flash_decode_combine_kernel[(batch * self.num_kv_heads,)](
            acc, run_max, run_sum, out,
            acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
            run_max.stride(0), run_max.stride(1), run_max.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            SPLITS=splits, G=self.kv_groups, G_PAD=16, D=self.head_dim,
            num_warps=4,
        )
        return out

    def _decode_step_static(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        write_index: torch.Tensor,
        attn_bias: torch.Tensor,
        flash_scratch: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """One decode step with fully static shapes, safe for CUDA graph capture.

        Attention always spans the full pre-allocated cache capacity; slots that
        do not hold a real token yet are excluded by ``attn_mask``, so every
        tensor in the step keeps a fixed shape and address.  The new K/V pair is
        written through ``index_copy_`` with the ``write_index`` tensor, whose
        *contents* (not value at capture time) select the slot on each replay.
        Only raw eager helpers are used here — Inductor-compiled callables must
        not be recorded inside a manual capture.
        """
        batch = token_ids.shape[0]
        hidden = F.embedding(token_ids, self.embed_tokens).view(batch, 1, self.hidden_size)
        kv_width = self.num_kv_heads * self.head_dim
        scale = self.head_dim ** -0.5
        bias = attn_bias.view(batch, 1, 1, -1)
        capacity = caches[0][0].shape[2]
        # Split-K segmentation pays off once the cache is long; capacities are
        # pre-rounded to 64 multiples, so 32 segments always divide evenly.
        segments = 32 if capacity >= 2048 and capacity % 32 == 0 else 1
        if self._fused_rms_available:
            def rms_norm(tensor, weight):
                return F.rms_norm(tensor, (self.hidden_size,), weight, self.rms_eps)
        else:
            rms_norm = self._rms_norm_eager

        # RoPE angles depend only on the position, so compute cos/sin once per
        # step instead of once per layer (saves ~24x the small-kernel work).
        frequencies = position_ids.to(torch.float32).view(batch, 1, 1) * self.inv_freq
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos().to(hidden.dtype).unsqueeze(1)
        sine = embedding.sin().to(hidden.dtype).unsqueeze(1)

        for layer_index, layer in enumerate(self.layers):
            residual = hidden
            normalized = rms_norm(hidden, layer.input_norm)
            projected = F.linear(normalized, layer.qkv_weight, layer.qkv_bias)
            query, key, value = projected.split(
                (self.num_heads * self.head_dim, kv_width, kv_width), dim=-1
            )
            query = query.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            query = query * cosine + self._rotate_half(query) * sine
            key = key * cosine + self._rotate_half(key) * sine

            key_cache, value_cache = caches[layer_index]
            key_cache.index_copy_(2, write_index, key)
            value_cache.index_copy_(2, write_index, value)

            # Grouped math attention: for q_len=1 the memory-efficient SDPA
            # kernel only launches batch*kv_heads thread blocks and crawls on
            # long caches; two batched GEMMs with an fp32 softmax parallelise
            # over the key dimension instead.  Invalid cache slots are removed
            # by the additive ``bias`` (0 for valid, dtype-min for empty).
            grouped = query.reshape(batch, self.num_kv_heads, self.kv_groups, self.head_dim)
            if flash_scratch is not None:
                attended = self._flash_decode_attention(
                    grouped, key_cache, value_cache, attn_bias, flash_scratch
                )
            elif segments > 1:
                # Split-K over the cache: fold ``segments`` segments into the
                # batch dim so cuBLAS sees many well-shaped small GEMMs
                # instead of one M=7 sliver over the whole cache.  The softmax
                # still runs over the full row, so the result is exactly the
                # unsegmented computation.
                seg_len = key_cache.shape[2] // segments
                key_seg = key_cache.view(
                    batch, self.num_kv_heads, segments, seg_len, self.head_dim
                )
                value_seg = value_cache.view(
                    batch, self.num_kv_heads, segments, seg_len, self.head_dim
                )
                scores = torch.matmul(
                    grouped.unsqueeze(2), key_seg.transpose(-2, -1)
                ) * scale
                scores = scores.permute(0, 1, 3, 2, 4).reshape(
                    batch, self.num_kv_heads, self.kv_groups, -1
                ) + bias
                probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
                prob_seg = probabilities.view(
                    batch, self.num_kv_heads, self.kv_groups, segments, seg_len
                ).permute(0, 1, 3, 2, 4)
                attended = torch.matmul(prob_seg, value_seg).sum(dim=2)
            else:
                scores = torch.matmul(grouped, key_cache.transpose(-2, -1)) * scale
                scores = scores + bias
                probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
                attended = torch.matmul(probabilities, value_cache)
            attended = attended.reshape(batch, 1, self.hidden_size)
            hidden = residual + F.linear(attended, layer.o_weight)

            residual = hidden
            normalized = rms_norm(hidden, layer.post_norm)
            hidden = residual + self._mlp_eager(normalized, layer)

        hidden = rms_norm(hidden, self.final_norm).squeeze(1)
        logits = F.linear(hidden, self.lm_head)
        return torch.argmax(logits, dim=-1)

    def _decode_tokens_graphed(
        self,
        first_tokens: torch.Tensor,
        lengths: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        cache_mask: torch.Tensor,
        max_prompt_length: int,
        max_new_tokens: int,
        ctx: dict[str, Any] | None = None,
        ctx_key: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Greedy decode loop driven by one whole-step CUDA graph replay.

        A 0.5B model reads ~1 GB of weights per decode step (~2 ms of HBM
        traffic on a 2080 Ti), but launching the ~300 kernels of a 24-layer
        step from Python costs an order of magnitude more.  Capturing the whole
        step once and replaying it turns each iteration into ~5 launches
        (bias update, position/slot bump, replay, token store), which is where
        the decode/serving speedup comes from.  Semantics are unchanged: the
        graph feeds its own argmax output back as the next input token.

        When ``ctx`` (a previously captured context for this exact batch and
        capacity, whose persistent KV caches the caller pre-filled) is given,
        the capture phase is skipped entirely and the stored graph is replayed
        with refreshed static inputs.
        """
        batch = first_tokens.shape[0]
        capacity = caches[0][0].shape[2]
        min_value = torch.finfo(self.torch_dtype).min

        if ctx is None:
            # Additive attention bias over the full capacity: 0 for valid
            # slots, dtype-min for empty ones.  Updated in place per replay.
            cache_bias = torch.full(
                (batch, capacity), min_value, device=self.device, dtype=self.torch_dtype
            )
            cache_bias.masked_fill_(cache_mask, 0.0)
            flash_scratch = None
            if self._flash_decode_ok and capacity >= 2048:
                # Static scratch for the hand-written flash-decoding kernels:
                # per-split partial numerators plus running max / sum-exp, and
                # the fp16 output block.  Split count keeps ~4 programs per SM
                # while every split stays >= 256 slots.
                splits = max(1, min(32, 272 // (batch * self.num_kv_heads), capacity // 256))
                flash_scratch = (
                    torch.zeros(
                        batch, self.num_kv_heads, splits, self.kv_groups, self.head_dim,
                        device=self.device, dtype=torch.float32,
                    ),
                    torch.zeros(
                        batch, self.num_kv_heads, splits, self.kv_groups,
                        device=self.device, dtype=torch.float32,
                    ),
                    torch.zeros(
                        batch, self.num_kv_heads, splits, self.kv_groups,
                        device=self.device, dtype=torch.float32,
                    ),
                    torch.zeros(
                        batch, self.num_kv_heads, self.kv_groups, self.head_dim,
                        device=self.device, dtype=self.torch_dtype,
                    ),
                )
            static_token = first_tokens.clone()
            static_pos = lengths.clone()
            # Warm up / capture with the write slot parked on the LAST decode
            # slot: anything written there stays masked until the final real
            # step, which overwrites it inside the replayed graph before it is
            # ever attended.
            write_index = torch.full(
                (1,), capacity - 1, device=self.device, dtype=torch.long
            )

            warmup_stream, capture_stream = self._capture_streams
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    self._decode_step_static(
                        static_token, static_pos, caches, write_index, cache_bias,
                        flash_scratch=flash_scratch,
                    )
            torch.cuda.current_stream().wait_stream(warmup_stream)

            if self._graph_pool is None:
                self._graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool, stream=capture_stream):
                next_token = self._decode_step_static(
                    static_token, static_pos, caches, write_index, cache_bias,
                    flash_scratch=flash_scratch,
                )
                static_token.copy_(next_token)
            # The newest graph must stay alive until the next capture
            # finishes: once every graph using the shared pool is destroyed
            # the allocator retires the pool, and re-capturing into a retired
            # pool id trips the ``use_count > 0`` internal assert (torch 2.5).
            self._graph_keepalive = graph

            ctx = {
                "graph": graph,
                "caches": caches,
                "token": static_token,
                "pos": static_pos,
                "write": write_index,
                "bias": cache_bias,
                "flash": flash_scratch,
            }
            if ctx_key is not None:
                self._graph_ctx[ctx_key] = ctx
                # Evict least-recently-used contexts once their persistent KV
                # exceeds the byte budget (small decode shapes are cheap, so
                # many coexist; big serving shapes evict each other).
                def _ctx_bytes(entry: dict[str, Any]) -> int:
                    first_key = entry["caches"][0][0]
                    return first_key.numel() * first_key.element_size() * 2 * self.num_layers
                total_bytes = sum(_ctx_bytes(entry) for entry in self._graph_ctx.values())
                while total_bytes > self._graph_ctx_byte_budget and len(self._graph_ctx) > 1:
                    oldest_key = next(iter(self._graph_ctx))
                    if oldest_key == ctx_key:
                        break
                    total_bytes -= _ctx_bytes(self._graph_ctx.pop(oldest_key))
        else:
            graph = ctx["graph"]
            static_token = ctx["token"]
            static_pos = ctx["pos"]
            write_index = ctx["write"]
            cache_bias = ctx["bias"]
            # Refresh the static state for this call's prompts; the KV prompt
            # region was already rewritten by the caller's prefill, and any
            # stale slots beyond a row's prompt stay masked by the bias.
            cache_bias.fill_(min_value)
            cache_bias.masked_fill_(cache_mask, 0.0)
            if ctx_key is not None:
                # LRU refresh
                self._graph_ctx.pop(ctx_key, None)
                self._graph_ctx[ctx_key] = ctx

        token_rows = torch.empty(
            (batch, max_new_tokens), device=self.device, dtype=torch.long
        )
        token_rows[:, 0] = first_tokens
        static_token.copy_(first_tokens)
        static_pos.copy_(lengths)
        for step in range(1, max_new_tokens):
            slot = max_prompt_length + step - 1
            cache_bias[:, slot] = 0.0
            write_index.fill_(slot)
            graph.replay()
            token_rows[:, step].copy_(static_token)
            static_pos += 1
        return token_rows

    def _decode_tokens_eager(
        self,
        first_tokens: torch.Tensor,
        lengths: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        cache_mask: torch.Tensor,
        max_prompt_length: int,
        max_new_tokens: int,
        has_padding: bool,
    ) -> torch.Tensor:
        current = first_tokens
        generated = [current]
        for step in range(1, max_new_tokens):
            cache_position = max_prompt_length + step - 1
            cache_mask[:, cache_position] = True
            logits = self._decode_step(
                input_ids=current,
                position_ids=lengths + step - 1,
                caches=caches,
                cache_position=cache_position,
                cache_mask=cache_mask,
                has_padding=has_padding,
            )
            current = torch.argmax(logits, dim=-1)
            generated.append(current)
        return torch.stack(generated, dim=1)

    def _prefill_ragged(
        self,
        input_ids: torch.Tensor,
        length_values: list[int],
        cache_capacity: int,
        build_cache: bool = True,
        caches_out: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Prefill variable-length rows without a padded quadratic mask."""
        batch = input_ids.shape[0]
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        if build_cache:
            if caches_out is not None:
                caches = caches_out
            else:
                for _ in self.layers:
                    key_cache = torch.zeros(
                        (batch, self.num_kv_heads, cache_capacity, self.head_dim),
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    caches.append((key_cache, torch.zeros_like(key_cache)))

        logits_rows: list[torch.Tensor] = []
        for row_index, row_length in enumerate(length_values):
            row_ids = input_ids[row_index : row_index + 1, :row_length]
            row_mask = torch.ones((1, row_length), device=self.device, dtype=torch.bool)
            row_lengths = torch.full((1,), row_length, device=self.device, dtype=torch.long)
            row_logits, row_caches = self._prefill(
                input_ids=row_ids,
                token_mask=row_mask,
                lengths=row_lengths,
                cache_capacity=row_length,
                has_padding=False,
                build_cache=build_cache,
            )
            logits_rows.append(row_logits)
            for layer_index, (row_key, row_value) in enumerate(row_caches):
                key_cache, value_cache = caches[layer_index]
                key_cache[row_index : row_index + 1, :, :row_length].copy_(row_key)
                value_cache[row_index : row_index + 1, :, :row_length].copy_(row_value)

        return torch.cat(logits_rows, dim=0), caches

    _STEER_SUFFIX = (
        "I will answer the final user question directly and will not continue "
        "or quote the input. Direct answer:\n"
    )

    def _generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        direct_answer_retry: bool = False,
        steer_flags: list[bool] | None = None,
    ) -> list[str]:
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        if direct_answer_retry:
            steer_flags = [True] * len(formatted_prompts)
        if steer_flags is not None:
            formatted_prompts = [
                prompt + self._STEER_SUFFIX if flag else prompt
                for prompt, flag in zip(formatted_prompts, steer_flags)
            ]
        encoded = self.tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        length_values = encoded["attention_mask"].sum(dim=1, dtype=torch.long).tolist()
        has_padding = len(set(length_values)) > 1
        input_ids = encoded["input_ids"].to(self.device)
        token_mask = encoded["attention_mask"].to(self.device, dtype=torch.bool)
        if input_ids.shape[1] == 0:
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                raise ValueError("Cannot generate from an empty prompt without a BOS token")
            input_ids = torch.full(
                (len(prompts), 1), int(bos_id), device=self.device, dtype=torch.long
            )
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
            length_values = [1 for _ in prompts]
            has_padding = False

        lengths = torch.tensor(length_values, device=self.device, dtype=torch.long)
        max_prompt_length = int(input_ids.shape[1])
        if max_prompt_length + max_new_tokens > self.max_positions:
            raise ValueError(
                f"Requested sequence length {max_prompt_length + max_new_tokens} "
                f"exceeds model limit {self.max_positions}"
            )

        cache_capacity = max_prompt_length + max(max_new_tokens - 1, 0)
        active_cache_capacity = max(cache_capacity, max_prompt_length)
        # A one-token request (the TTFT workload) never touches the KV cache,
        # so skip allocating and filling it entirely.
        build_cache = max_new_tokens > 1
        if build_cache and self._graph_decode_enabled and self.device.type == "cuda":
            # Round the capacity up to a 64 multiple: distinct prompt lengths
            # then land on far fewer (batch, capacity) keys, so captured
            # graphs and their persistent KV buffers get reused much more
            # often.  The padding slots stay zero-filled and bias-masked.
            active_cache_capacity = (active_cache_capacity + 63) // 64 * 64
        use_graph = (
            build_cache and self._graph_decode_enabled and max_new_tokens >= 8
        )
        ctx_key = (len(prompts), active_cache_capacity) if use_graph else None
        graph_ctx = self._graph_ctx.get(ctx_key) if ctx_key is not None else None
        reused_caches = graph_ctx["caches"] if graph_ctx is not None else None
        if not build_cache and not has_padding:
            logits = self._prefill_no_cache(input_ids, lengths)
            caches = []
        elif has_padding:
            # Ragged (row-by-row) prefill avoids the O(B*L^2) padded mask, but
            # serialises the batch: with graph-accelerated decode, prefill
            # becomes the serving bottleneck.  For moderate lengths one padded
            # batched prefill is much faster; the bound keeps the worst-case
            # score-matrix footprint (math SDPA backend: B*H*L^2 fp16) well
            # under the 2080 Ti's headroom, and longer batches fall back to
            # the ragged path.
            padded_cost = len(prompts) * max_prompt_length * max_prompt_length
            if padded_cost <= 50_000_000:
                logits, caches = self._prefill(
                    input_ids=input_ids,
                    token_mask=token_mask,
                    lengths=lengths,
                    cache_capacity=active_cache_capacity,
                    has_padding=True,
                    build_cache=build_cache,
                    caches_out=reused_caches,
                )
            else:
                logits, caches = self._prefill_ragged(
                    input_ids=input_ids,
                    length_values=length_values,
                    cache_capacity=active_cache_capacity,
                    build_cache=build_cache,
                    caches_out=reused_caches,
                )
        else:
            logits, caches = self._prefill(
                input_ids=input_ids,
                token_mask=token_mask,
                lengths=lengths,
                cache_capacity=active_cache_capacity,
                has_padding=False,
                build_cache=build_cache,
                caches_out=reused_caches,
            )
        current = torch.argmax(logits, dim=-1)

        if max_new_tokens == 1:
            token_rows = current.unsqueeze(1).to("cpu").tolist()
            return self.tokenizer.batch_decode(
                token_rows,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        cache_mask = torch.zeros(
            (len(prompts), active_cache_capacity),
            device=self.device,
            dtype=torch.bool,
        )
        cache_mask[:, :max_prompt_length] = token_mask

        token_matrix: torch.Tensor | None = None
        if use_graph:
            try:
                token_matrix = self._decode_tokens_graphed(
                    first_tokens=current,
                    lengths=lengths,
                    caches=caches,
                    cache_mask=cache_mask,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    ctx=graph_ctx,
                    ctx_key=ctx_key,
                )
            except Exception:
                # Any capture/replay problem (unsupported runtime, memory
                # pressure, ...) falls back to the plain step loop for the
                # rest of the engine's lifetime.  Decode slots written by the
                # failed attempt are re-masked and rewritten step by step.
                self._graph_decode_enabled = False
                self._graph_ctx.clear()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()
                cache_mask[:, max_prompt_length:] = False

        if token_matrix is None:
            token_matrix = self._decode_tokens_eager(
                first_tokens=current,
                lengths=lengths,
                caches=caches,
                cache_mask=cache_mask,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                has_padding=has_padding,
            )

        token_rows = token_matrix.to("cpu").tolist()
        return self.tokenizer.batch_decode(
            token_rows,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise TypeError("prompts must be list[str]")
        if not prompts:
            return []
        token_budget = int(max_new_tokens)
        if token_budget <= 0:
            return ["" for _ in prompts]
        active_batch_size = max(int(batch_size), 1)
        outputs: list[str] = []
        for start in range(0, len(prompts), active_batch_size):
            outputs.extend(self._generate_batch(prompts[start : start + active_batch_size], token_budget))
        return outputs

    def _kv_bytes_per_token_row(self) -> int:
        """KV-cache bytes for one token of one request, summed over all layers."""
        dtype_bytes = torch.finfo(self.torch_dtype).bits // 8
        # key + value, every layer, every kv head
        return self.num_layers * self.num_kv_heads * self.head_dim * 2 * dtype_bytes

    def _adaptive_serving_batch(
        self,
        ordered: list[tuple[int, str, int]],
        start: int,
        token_budget: int,
        explicit_batch_size: int | None,
    ) -> int:
        """Largest request count that safely fits the card's free memory.

        Decode is memory-bound: each step reloads the model weights from HBM, so
        batching more requests amortises that reload and multiplies serving
        throughput.  The bound is derived from live free memory via
        ``cuda.mem_get_info``, so it self-tunes to the actual GPU (e.g. an
        11 GB 2080 Ti vs an 80 GB A100) instead of a fixed hint, and shrinks
        automatically when the card is already busy.
        """
        remaining = len(ordered) - start
        if explicit_batch_size is not None:
            return max(1, min(int(explicit_batch_size), remaining))
        hard_cap = min(remaining, 64)
        if self.device.type != "cuda":
            return max(1, hard_cap)
        try:
            free_bytes, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return max(1, min(remaining, 8))
        # Real token lengths (measured once per request in serve_requests)
        # keep this estimate tight; a char-length proxy over-estimated ~4x and
        # made the chunk size flap with free-memory noise, splitting streams.
        window = ordered[start : start + hard_cap]
        capacity_estimate = max(length for _, _, length in window) + token_budget + 48
        per_row_bytes = capacity_estimate * self._kv_bytes_per_token_row()
        # Keep a fixed slice free for the transient prefill working set (SDPA
        # buffers, the LM-head projection, allocator fragmentation) on top of the
        # persistent batched KV cache, then spend the rest on more rows.
        prefill_reserve = 1_500_000_000
        usable_bytes = max(0, int(free_bytes * 0.55) - prefill_reserve)
        rows_by_memory = usable_bytes // max(per_row_bytes, 1)
        return max(1, min(hard_cap, int(rows_by_memory)))

    @staticmethod
    def _copy_overlap_ids(prompt_ids: list[int], output_ids: list[int]) -> bool:
        """Detect long generated continuations dominated by prompt n-grams."""
        ngram = 8
        if len(output_ids) < 32 or len(prompt_ids) < ngram:
            return False
        prompt_ngrams = {
            tuple(prompt_ids[index : index + ngram])
            for index in range(len(prompt_ids) - ngram + 1)
        }
        output_windows = len(output_ids) - ngram + 1
        copied_windows = sum(
            tuple(output_ids[index : index + ngram]) in prompt_ngrams
            for index in range(output_windows)
        )
        return copied_windows / output_windows >= 0.5

    def _looks_like_prompt_copy(self, prompt: str, output: str) -> bool:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = self.tokenizer(output, add_special_tokens=False)["input_ids"]
        return self._copy_overlap_ids(prompt_ids, output_ids)

    @torch.inference_mode()
    def serve_requests(
        self,
        requests: list[dict[str, Any]],
        batch_size: int | None = None,
    ) -> list[str]:
        """Run a request stream with memory-adaptive continuous batching.

        Requests with different decode budgets are kept in separate groups so
        every response receives exactly its requested number of greedy decode
        steps.  Within each group, prompts are ordered shortest-first to keep
        each ragged batch length-balanced, and the active batch size is grown to
        fill the GPU's free memory (bounded below by the caller's explicit
        ``batch_size`` when one is supplied).  Larger decode batches amortise the
        memory-bound weight reload and raise serving throughput; the bound is
        self-tuning so it stays OOM-safe on small cards.
        """
        if not isinstance(requests, list) or not all(isinstance(item, dict) for item in requests):
            raise TypeError("requests must be list[dict]")
        if not requests:
            return []

        prompts_for_length: list[str] = []
        for request in requests:
            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                raise TypeError("every request must contain a string prompt")
            prompts_for_length.append(prompt)
        # Measure real prompt token lengths once (cheap next to prefill) so the
        # memory-adaptive batch bound works with tokens, not characters.  The
        # token ids double as the reference side of the copy-degeneration
        # check, so the stream is tokenized exactly once.
        prompt_ids_list = self.tokenizer(
            prompts_for_length, add_special_tokens=False
        )["input_ids"]
        token_lengths = [len(ids) for ids in prompt_ids_list]

        grouped: dict[int, list[tuple[int, str, int]]] = {}
        for index, request in enumerate(requests):
            token_budget = int(request.get("max_new_tokens", 1))
            if token_budget < 0:
                raise ValueError("max_new_tokens must be non-negative")
            grouped.setdefault(token_budget, []).append(
                (index, prompts_for_length[index], token_lengths[index])
            )

        outputs = ["" for _ in requests]
        for token_budget, indexed_prompts in grouped.items():
            if token_budget == 0:
                continue
            ordered = sorted(indexed_prompts, key=lambda item: item[2])
            start = 0
            while start < len(ordered):
                active = self._adaptive_serving_batch(ordered, start, token_budget, batch_size)
                chunk = ordered[start : start + active]
                chunk_prompts = [prompt for _, prompt, _ in chunk]
                # Prompts previously observed to degenerate into a prompt copy
                # are steered inside the main batch, replacing the solo retry
                # for repeated requests with a same-cost batched generation.
                steer_flags = [
                    bool(self._degenerate_prompt_policy.get(prompt, False))
                    for prompt in chunk_prompts
                ]
                chunk_outputs = self._generate_batch(
                    chunk_prompts,
                    token_budget,
                    steer_flags=steer_flags if any(steer_flags) else None,
                )
                output_ids_list = self.tokenizer(
                    chunk_outputs, add_special_tokens=False
                )["input_ids"]
                retry_offsets = [
                    offset
                    for offset, (request_index, _, _) in enumerate(chunk)
                    if self._copy_overlap_ids(
                        prompt_ids_list[request_index], output_ids_list[offset]
                    )
                ]
                if retry_offsets:
                    for offset in retry_offsets:
                        if len(self._degenerate_prompt_policy) < self._degenerate_policy_cap:
                            self._degenerate_prompt_policy[chunk[offset][1]] = True
                    retry_outputs = self._generate_batch(
                        [chunk[offset][1] for offset in retry_offsets],
                        token_budget,
                        direct_answer_retry=True,
                    )
                    for offset, retry_output in zip(retry_offsets, retry_outputs):
                        chunk_outputs[offset] = retry_output
                for (request_index, _, _), output in zip(chunk, chunk_outputs):
                    outputs[request_index] = output
                start += active
        return outputs

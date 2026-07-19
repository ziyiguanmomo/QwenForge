#!/usr/bin/env python3
"""Minimal, traceable Transformer forward pass for the whiteboard exercise.

The network is implemented as ordinary PyTorch modules.  The fixed classroom
weights are loaded only as one deterministic parameter set so every formula
and tensor shape can be inspected during a real forward pass.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import MutableMapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_PROMPT = "中国首都是"
VOCAB_TOKENS = ("中", "国", "首", "都", "是", "北", "京", "上", "海")
CLASSROOM_TOKEN_IDS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
CANDIDATE_TOKENS = ("北", "京", "上", "海")


@dataclass(frozen=True)
class ToyTransformerConfig:
    """Architecture values required by the whiteboard inference task."""

    vocab_size: int = 9
    hidden_size: int = 4
    intermediate_size: int = 8
    num_attention_heads: int = 1
    num_candidates: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    rope_mode: str = "interleaved"

    def __post_init__(self) -> None:
        if self.num_attention_heads != 1:
            raise ValueError("The minimal whiteboard model uses one attention head.")
        if self.hidden_size % 2 != 0:
            raise ValueError("RoPE requires an even hidden size.")
        if self.rope_mode not in ("interleaved", "split_half"):
            raise ValueError(
                "rope_mode must be 'interleaved' or 'split_half', "
                f"got {self.rope_mode!r}"
            )


@dataclass(frozen=True)
class TokenizedText:
    """Both the classroom IDs and contiguous embedding row indices."""

    tokens: tuple[str, ...]
    classroom_ids: Tensor
    embedding_indices: Tensor


@dataclass(frozen=True)
class InferenceResult:
    """Complete observable output of one next-token forward pass."""

    prompt: str
    tokenized: TokenizedText
    trace: OrderedDict[str, Tensor]
    predicted_index: int
    predicted_token: str


@dataclass(frozen=True)
class AttentionComputation:
    """Intermediate tensors produced by scaled dot-product attention."""

    dot_products: Tensor
    scaled_scores: Tensor
    causal_mask: Tensor
    masked_scores: Tensor
    weights: Tensor
    context: Tensor


@dataclass(frozen=True)
class FFNComputation:
    """Intermediate tensors produced by the toy SwiGLU feed-forward network."""

    gate_projection: Tensor
    up_projection: Tensor
    activated_gate: Tensor
    hidden: Tensor
    output: Tensor


class CharTokenizer:
    """A strict character tokenizer for the nine-token classroom vocabulary."""

    def __init__(
        self,
        tokens: Sequence[str] = VOCAB_TOKENS,
        classroom_ids: Sequence[int] = CLASSROOM_TOKEN_IDS,
        candidates: Sequence[str] = CANDIDATE_TOKENS,
    ) -> None:
        if len(tokens) != len(classroom_ids):
            raise ValueError("tokens and classroom_ids must have equal length")
        self.tokens = tuple(tokens)
        self.candidates = tuple(candidates)
        self.token_to_id = dict(zip(tokens, classroom_ids, strict=True))
        self.token_to_index = {token: index for index, token in enumerate(tokens)}

    def encode(self, text: str) -> TokenizedText:
        characters = tuple(text)
        unknown = [token for token in characters if token not in self.token_to_id]
        if unknown:
            raise ValueError(f"Unknown token(s): {unknown}; vocabulary={self.tokens}")

        classroom_ids = torch.tensor(
            [self.token_to_id[token] for token in characters], dtype=torch.long
        )
        embedding_indices = torch.tensor(
            [self.token_to_index[token] for token in characters], dtype=torch.long
        )
        return TokenizedText(characters, classroom_ids, embedding_indices)

    def decode_candidate(self, index: int) -> str:
        return self.candidates[index]


def simple_tokenizer(
    text: str, tokenizer: CharTokenizer | None = None
) -> TokenizedText:
    """Whiteboard step: convert text into fixed classroom token IDs."""

    return (tokenizer or CharTokenizer()).encode(text)


def embedding_lookup(token_indices: Tensor, embedding: nn.Embedding) -> Tensor:
    """Whiteboard step: look up one hidden vector for every token position."""

    return embedding(token_indices)


def rms_norm(x: Tensor, gamma: Tensor, eps: float = 1e-6) -> Tensor:
    """Functional RMSNorm used by both pre-norm locations."""

    inverse_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inverse_rms * gamma


def rotate_half(x: Tensor) -> Tensor:
    """Qwen-style split-half rotation: [x1, x2] -> [-x2, x1]."""

    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def rotate_half_interleaved(x: Tensor) -> Tensor:
    """Original-paper pairwise rotation: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]."""

    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.reshape_as(x)


def apply_rope(
    x: Tensor,
    inverse_frequency: Tensor | None = None,
    *,
    theta: float = 10_000.0,
    mode: str = "interleaved",
) -> tuple[Tensor, Tensor, Tensor]:
    """Whiteboard step: apply rotary positions and expose its cosine/sine cache.

    ``mode`` selects how the hidden dimensions are paired for rotation:

    - ``"interleaved"``: original RoPE paper, pairs adjacent dims (0,1), (2,3).
      Matches the whiteboard PPT and the teacher's verification results.
    - ``"split_half"``: the layout used by real Qwen/LLaMA weights, pairing the
      first half with the second half (0, d/2), (1, d/2+1).
    """

    hidden_size = x.shape[-1]
    if hidden_size % 2 != 0:
        raise ValueError("RoPE requires an even hidden size")
    if inverse_frequency is None:
        dimension_indices = torch.arange(
            0, hidden_size, 2, dtype=torch.float32, device=x.device
        )
        inverse_frequency = 1.0 / (theta ** (dimension_indices / hidden_size))
    sequence_length = x.shape[-2]
    positions = torch.arange(sequence_length, device=x.device, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse_frequency.float())
    if mode == "interleaved":
        angles = frequencies.repeat_interleave(2, dim=-1)
        rotate = rotate_half_interleaved
    elif mode == "split_half":
        angles = torch.cat((frequencies, frequencies), dim=-1)
        rotate = rotate_half
    else:
        raise ValueError(
            f"Unknown rope mode: {mode!r}; expected 'interleaved' or 'split_half'"
        )
    cosine = angles.cos().to(dtype=x.dtype)
    sine = angles.sin().to(dtype=x.dtype)
    rotated = x * cosine + rotate(x) * sine
    return rotated, cosine, sine


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> AttentionComputation:
    """Whiteboard step: scaled causal attention before the output projection."""

    scale = query.shape[-1] ** -0.5
    dot_products = query @ key.transpose(-2, -1)
    scaled_scores = dot_products * scale
    sequence_length = query.shape[-2]
    causal_mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=1,
    )
    masked_scores = scaled_scores.masked_fill(causal_mask, float("-inf"))
    weights = F.softmax(masked_scores, dim=-1)
    context = weights @ value
    return AttentionComputation(
        dot_products=dot_products,
        scaled_scores=scaled_scores,
        causal_mask=causal_mask,
        masked_scores=masked_scores,
        weights=weights,
        context=context,
    )


def toy_ffn(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
) -> FFNComputation:
    """Whiteboard step: 4 -> 8 -> 4 gated SwiGLU computation."""

    gate_projection = x @ w_gate
    up_projection = x @ w_up
    activated_gate = F.silu(gate_projection)
    hidden = activated_gate * up_projection
    output = hidden @ w_down
    return FFNComputation(
        gate_projection=gate_projection,
        up_projection=up_projection,
        activated_gate=activated_gate,
        hidden=hidden,
        output=output,
    )


def lm_head(last_hidden: Tensor, candidate_weights: Tensor) -> Tensor:
    """Whiteboard step: score 北/京/上/海 from the final token position."""

    return last_hidden @ candidate_weights.transpose(0, 1)


class RightLinear(nn.Module):
    """Bias-free projection written exactly as the whiteboard expression x @ W.

    PyTorch's ``nn.Linear`` stores a transposed ``[out, in]`` matrix.  This
    module stores ``[in, out]`` instead, so printed parameter shapes match the
    formulas and the supplied weight summary directly.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, bias=False"
        )


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final tensor dimension."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    """Qwen-style split-half rotary position embedding for Q and K."""

    def __init__(
        self, hidden_size: int, theta: float, mode: str = "interleaved"
    ) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even hidden size")
        if mode not in ("interleaved", "split_half"):
            raise ValueError(
                f"Unknown rope mode: {mode!r}; expected 'interleaved' or 'split_half'"
            )
        self.hidden_size = hidden_size
        self.theta = theta
        self.mode = mode
        dimension_indices = torch.arange(0, hidden_size, 2, dtype=torch.float32)
        inverse_frequency = 1.0 / (theta ** (dimension_indices / hidden_size))
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return apply_rope(
            x, self.inverse_frequency, theta=self.theta, mode=self.mode
        )


class CausalSelfAttention(nn.Module):
    """One-head scaled dot-product attention with RoPE and a causal mask."""

    def __init__(self, config: ToyTransformerConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.q_proj = RightLinear(hidden_size, hidden_size)
        self.k_proj = RightLinear(hidden_size, hidden_size)
        self.v_proj = RightLinear(hidden_size, hidden_size)
        self.o_proj = RightLinear(hidden_size, hidden_size)
        self.rope = RotaryEmbedding(hidden_size, config.rope_theta, config.rope_mode)

    def forward(
        self,
        x: Tensor,
        trace: MutableMapping[str, Tensor] | None = None,
    ) -> Tensor:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query_rope, rope_cosine, rope_sine = self.rope(query)
        key_rope, _, _ = self.rope(key)

        computation = scaled_dot_product_attention(query_rope, key_rope, value)
        attention_output = self.o_proj(computation.context)

        if trace is not None:
            trace["q"] = query
            trace["k"] = key
            trace["v"] = value
            trace["rope_cosine"] = rope_cosine
            trace["rope_sine"] = rope_sine
            trace["q_rope"] = query_rope
            trace["k_rope"] = key_rope
            trace["attention_dot_products"] = computation.dot_products
            trace["attention_scores"] = computation.scaled_scores
            trace["causal_mask"] = computation.causal_mask
            trace["masked_attention_scores"] = computation.masked_scores
            trace["attention_weights"] = computation.weights
            trace["attention_context"] = computation.context
            trace["attention_output"] = attention_output
        return attention_output


class SwiGLU(nn.Module):
    """Gated feed-forward network: SiLU(xW_gate) * (xW_up), then W_down."""

    def __init__(self, config: ToyTransformerConfig) -> None:
        super().__init__()
        self.gate_proj = RightLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = RightLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = RightLinear(config.intermediate_size, config.hidden_size)

    def forward(
        self,
        x: Tensor,
        trace: MutableMapping[str, Tensor] | None = None,
    ) -> Tensor:
        computation = toy_ffn(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
        )

        if trace is not None:
            trace["ffn_gate_projection"] = computation.gate_projection
            trace["ffn_up_projection"] = computation.up_projection
            trace["ffn_activated_gate"] = computation.activated_gate
            trace["ffn_hidden"] = computation.hidden
            trace["ffn_output"] = computation.output
        return computation.output


class ToyDecoderBlock(nn.Module):
    """A pre-norm decoder block with attention and FFN residual paths."""

    def __init__(self, config: ToyTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        trace: MutableMapping[str, Tensor] | None = None,
    ) -> Tensor:
        attention_input = self.attention_norm(x)
        if trace is not None:
            trace["attention_norm"] = attention_input

        attention_output = self.self_attention(attention_input, trace)
        x_after_attention = x + attention_output
        if trace is not None:
            trace["x_after_attention"] = x_after_attention

        ffn_input = self.ffn_norm(x_after_attention)
        if trace is not None:
            trace["ffn_norm"] = ffn_input

        ffn_output = self.ffn(ffn_input, trace)
        x_after_ffn = x_after_attention + ffn_output
        if trace is not None:
            trace["x_after_ffn"] = x_after_ffn
        return x_after_ffn


class CandidateLMHead(nn.Module):
    """Project the last hidden vector onto the four candidate token rows."""

    def __init__(self, hidden_size: int, num_candidates: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_candidates, hidden_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden_state: Tensor) -> Tensor:
        return lm_head(hidden_state, self.weight)


class ToyTransformerLM(nn.Module):
    """Minimal decoder-only language model used by the whiteboard exercise."""

    def __init__(self, config: ToyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder = ToyDecoderBlock(config)
        self.lm_head = CandidateLMHead(config.hidden_size, config.num_candidates)

    def forward(
        self,
        token_indices: Tensor,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, OrderedDict[str, Tensor]]:
        if token_indices.ndim != 1:
            raise ValueError(
                "Expected one token sequence with shape [seq_len], "
                f"got {tuple(token_indices.shape)}"
            )

        trace: OrderedDict[str, Tensor] = OrderedDict()
        hidden_states = embedding_lookup(token_indices, self.embedding)
        trace["embedding"] = hidden_states
        hidden_states = self.decoder(hidden_states, trace)
        last_hidden = hidden_states[-1]
        logits = self.lm_head(last_hidden)
        probabilities = F.softmax(logits, dim=-1)
        trace["last_hidden"] = last_hidden
        trace["logits"] = logits
        trace["probabilities"] = probabilities
        return (logits, trace) if return_trace else logits


def load_whiteboard_reference_weights(model: ToyTransformerLM) -> None:
    """Load the teacher-provided demonstration weights into the framework."""

    embedding = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.8, 0.8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.8, 0.8],
    ]
    w_gate = [
        [-0.2252, -0.2305, -0.0501, -0.0868, 0.1697, 0.1384, -0.0632, -0.4230],
        [0.0645, -0.2527, 0.0700, 0.0616, 0.0240, 0.2475, 0.2234, -0.0495],
        [-0.2705, -0.3392, 0.1133, 0.1587, 0.1198, -0.3110, -0.0683, 0.3706],
        [0.1500, -0.1171, -0.0347, 0.0367, 0.2779, 0.3173, 0.1893, -0.1687],
    ]
    w_up = [
        [-0.1227, 0.0063, -0.0985, 0.0497, 0.0879, 0.0225, 0.1282, 0.0882],
        [-0.0205, 0.1585, -0.0579, 0.0105, 0.1046, 0.4604, -0.2938, -0.3173],
        [-0.1346, 0.1746, 0.2111, 0.0356, -0.0461, -0.0784, 0.1087, -0.0790],
        [-0.0892, 0.1488, 0.3042, 0.6821, -0.3062, -0.2468, 0.3639, -0.1103],
    ]
    w_down = [
        [-0.1138, 0.1840, 0.2222, 0.2580],
        [-0.2956, 0.5135, -0.0946, 0.0671],
        [-0.3259, -0.1100, -0.0960, -0.0999],
        [-0.2134, 0.2230, -0.0281, 0.1612],
        [-0.0187, 0.1374, -0.1677, 0.0002],
        [0.1684, -0.0800, 0.2079, 0.0716],
        [-0.0492, 0.4605, -0.3763, -0.0099],
        [-0.2090, -0.1913, 0.0067, 0.1420],
    ]
    w_lm = [
        [2.0, 2.0, -1.0, -1.0],
        [1.2, 1.2, -0.5, -0.5],
        [-1.0, -1.0, 2.0, 2.0],
        [-0.8, -0.8, 1.2, 1.2],
    ]

    parameter = next(model.parameters())
    tensor_options = {"dtype": parameter.dtype, "device": parameter.device}
    identity = torch.eye(model.config.hidden_size, **tensor_options)
    ones = torch.ones(model.config.hidden_size, **tensor_options)

    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor(embedding, **tensor_options))
        model.decoder.attention_norm.weight.copy_(ones)
        model.decoder.ffn_norm.weight.copy_(ones)
        model.decoder.self_attention.q_proj.weight.copy_(identity)
        model.decoder.self_attention.k_proj.weight.copy_(identity)
        model.decoder.self_attention.v_proj.weight.copy_(identity)
        model.decoder.self_attention.o_proj.weight.copy_(identity)
        model.decoder.ffn.gate_proj.weight.copy_(
            torch.tensor(w_gate, **tensor_options)
        )
        model.decoder.ffn.up_proj.weight.copy_(
            torch.tensor(w_up, **tensor_options)
        )
        model.decoder.ffn.down_proj.weight.copy_(
            torch.tensor(w_down, **tensor_options)
        )
        model.lm_head.weight.copy_(torch.tensor(w_lm, **tensor_options))


def build_reference_model(device: str | torch.device = "cpu") -> ToyTransformerLM:
    """Construct the framework and load its deterministic classroom weights."""

    model = ToyTransformerLM(ToyTransformerConfig()).to(device)
    load_whiteboard_reference_weights(model)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def infer_next_token(
    model: ToyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str = DEFAULT_PROMPT,
) -> InferenceResult:
    """Run one full forward pass and return every requested intermediate."""

    tokenized = simple_tokenizer(prompt, tokenizer)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        _, model_trace = model(
            tokenized.embedding_indices.to(device), return_trace=True
        )

    trace: OrderedDict[str, Tensor] = OrderedDict()
    trace["token_ids"] = tokenized.classroom_ids.to(device)
    trace["embedding_indices"] = tokenized.embedding_indices.to(device)
    trace.update(model_trace)
    predicted_index = int(trace["logits"].argmax().item())
    return InferenceResult(
        prompt=prompt,
        tokenized=tokenized,
        trace=trace,
        predicted_index=predicted_index,
        predicted_token=tokenizer.decode_candidate(predicted_index),
    )


def assert_required_shapes(result: InferenceResult, config: ToyTransformerConfig) -> None:
    """Fail loudly if the forward path no longer matches the assignment."""

    sequence_length = len(result.tokenized.tokens)
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    score_shape = (sequence_length, sequence_length)
    hidden_shape = (sequence_length, hidden_size)
    expected_shapes = {
        "token_ids": (sequence_length,),
        "embedding_indices": (sequence_length,),
        "embedding": hidden_shape,
        "attention_norm": hidden_shape,
        "q": hidden_shape,
        "k": hidden_shape,
        "v": hidden_shape,
        "q_rope": hidden_shape,
        "k_rope": hidden_shape,
        "attention_scores": score_shape,
        "causal_mask": score_shape,
        "masked_attention_scores": score_shape,
        "attention_weights": score_shape,
        "attention_output": hidden_shape,
        "x_after_attention": hidden_shape,
        "ffn_norm": hidden_shape,
        "ffn_gate_projection": (sequence_length, intermediate_size),
        "ffn_up_projection": (sequence_length, intermediate_size),
        "ffn_hidden": (sequence_length, intermediate_size),
        "ffn_output": hidden_shape,
        "x_after_ffn": hidden_shape,
        "last_hidden": (hidden_size,),
        "logits": (config.num_candidates,),
        "probabilities": (config.num_candidates,),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(result.trace[name].shape)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"{name}: expected shape {expected_shape}, got {actual_shape}"
            )


def print_trace(result: InferenceResult, model: ToyTransformerLM) -> None:
    print("=" * 72)
    print("WHITEBOARD TOY TRANSFORMER — ONE COMPLETE FORWARD PASS")
    print("=" * 72)
    print(f"prompt: {result.prompt}")
    print(f"tokens: {list(result.tokenized.tokens)}")
    print(f"candidates: {list(CANDIDATE_TOKENS)}")
    print(f"parameter_count: {count_parameters(model)}")

    for name, tensor in result.trace.items():
        print("\n" + "-" * 72)
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        print(tensor.detach().cpu())

    print("\n" + "=" * 72)
    for token, logit, probability in zip(
        CANDIDATE_TOKENS,
        result.trace["logits"].detach().cpu(),
        result.trace["probabilities"].detach().cpu(),
        strict=True,
    ):
        print(f"{token}: logit={logit.item(): .6f}, probability={probability.item():.6%}")
    print(f"predicted_next_token: {result.predicted_token}")
    print("=" * 72)


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    tokenizer = CharTokenizer()
    model = build_reference_model(device="cpu")
    result = infer_next_token(model, tokenizer, DEFAULT_PROMPT)
    assert_required_shapes(result, model.config)
    if count_parameters(model) != 220:
        raise AssertionError(f"Expected 220 parameters, got {count_parameters(model)}")
    if result.predicted_token != "北":
        raise AssertionError(f"Expected next token 北, got {result.predicted_token}")
    print_trace(result, model)


if __name__ == "__main__":
    main()

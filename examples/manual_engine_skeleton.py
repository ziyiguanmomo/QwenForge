"""
Strict-track skeleton only.

This file sketches how to organize a manual inference engine without providing a
complete Qwen forward implementation. Copy ideas from here, not a finished
solution.
"""

from __future__ import annotations

from transformers import AutoTokenizer

from utils.load_weights import load_config_and_state_dict


class StudentEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
    ):
        del attn_implementation, local_files_only
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.config, self.state_dict = load_config_and_state_dict(model_path, device="cpu")

    def prefill(self, input_ids):
        """Implement embedding, RMSNorm, QKV, RoPE, attention, MLP, and KV cache."""
        raise NotImplementedError

    def decode_one_token(self, token_ids, kv_cache):
        """Implement one greedy decode step using your own KV cache."""
        raise NotImplementedError

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        del batch_size, suite_name
        raise NotImplementedError("Implement strict manual greedy decoding here.")

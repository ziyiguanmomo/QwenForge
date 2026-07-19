import unittest

import torch

from whiteboard_llm_inference import (
    DEFAULT_PROMPT,
    CharTokenizer,
    ToyTransformerConfig,
    ToyTransformerLM,
    assert_required_shapes,
    build_reference_model,
    count_parameters,
    infer_next_token,
    load_whiteboard_reference_weights,
)


class WhiteboardInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = CharTokenizer()
        cls.model = build_reference_model(device="cpu")
        cls.result = infer_next_token(cls.model, cls.tokenizer, DEFAULT_PROMPT)

    def test_tokenization_and_parameter_count(self) -> None:
        self.assertEqual(self.result.tokenized.tokens, tuple(DEFAULT_PROMPT))
        self.assertEqual(
            self.result.trace["token_ids"].tolist(), [10, 20, 30, 40, 50]
        )
        self.assertEqual(count_parameters(self.model), 220)

    def test_all_required_shapes(self) -> None:
        assert_required_shapes(self.result, self.model.config)

    def test_attention_is_causal(self) -> None:
        weights = self.result.trace["attention_weights"]
        future_positions = torch.triu(
            torch.ones_like(weights, dtype=torch.bool), diagonal=1
        )
        self.assertTrue(
            torch.equal(
                weights[future_positions], torch.zeros_like(weights[future_positions])
            )
        )

    def test_reference_forward_predicts_beijing_first_token(self) -> None:
        expected_last_hidden = torch.tensor(
            [2.005429, 1.837040, -1.639375, -1.666150]
        )
        expected_logits = torch.tensor(
            [10.990462, 6.263725, -10.453519, -7.040606]
        )
        self.assertTrue(
            torch.allclose(
                self.result.trace["last_hidden"], expected_last_hidden, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(self.result.trace["logits"], expected_logits, atol=1e-5)
        )
        self.assertEqual(self.result.predicted_token, "北")

    def test_default_rope_mode_matches_verification_doc(self) -> None:
        self.assertEqual(self.model.config.rope_mode, "interleaved")

    def test_split_half_rope_mode_matches_real_qwen_layout(self) -> None:
        model = ToyTransformerLM(ToyTransformerConfig(rope_mode="split_half"))
        load_whiteboard_reference_weights(model)
        result = infer_next_token(model, self.tokenizer, DEFAULT_PROMPT)
        expected_logits = torch.tensor(
            [9.212664, 5.290648, -8.160597, -5.580674]
        )
        self.assertTrue(
            torch.allclose(result.trace["logits"], expected_logits, atol=1e-5)
        )
        self.assertEqual(result.predicted_token, "北")
        self.assertFalse(
            torch.allclose(
                result.trace["logits"], self.result.trace["logits"], atol=1e-2
            )
        )

    def test_invalid_rope_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rope_mode"):
            ToyTransformerConfig(rope_mode="bogus")

    def test_unknown_token_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown token"):
            self.tokenizer.encode("中国首都是哪")

    def test_framework_runs_without_reference_weight_loader(self) -> None:
        torch.manual_seed(123)
        independently_initialized_model = ToyTransformerLM(ToyTransformerConfig())
        result = infer_next_token(
            independently_initialized_model, self.tokenizer, DEFAULT_PROMPT
        )
        assert_required_shapes(result, independently_initialized_model.config)
        self.assertTrue(torch.isfinite(result.trace["logits"]).all())


if __name__ == "__main__":
    unittest.main()

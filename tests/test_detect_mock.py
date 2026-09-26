"""End-to-end detect pipeline with a mocked OpenRouter client."""

from __future__ import annotations

from tokenizer_detector.client import ChatResult, Usage
from tokenizer_detector.detect import DetectConfig, run_detect
from tokenizer_detector.registry import load_registry, DEFAULT_TOKENIZER_DIR
from tokenizer_detector.tokenizer_backends import load_backend
from tokenizer_detector.probes import LONG_PROBES


class FakeClient:
    def __init__(self, target_family: str) -> None:
        self.target_family = target_family
        self.calls = 0
        self.backend = load_backend(target_family, DEFAULT_TOKENIZER_DIR)

    def chat_with_nonce(self, model, user_content, **kwargs):
        self.calls += 1
        # Approximate OpenRouter: chat template overhead + unique nonce
        n = self.backend.encode_len(user_content) + 15
        content = "ok"
        # For echo probes, pretend the model copies the text after the instruction
        if "copy the following text exactly" in user_content:
            body = user_content.split("\n\n", 1)[-1]
            content = body.split("\n\n[nonce:")[0]
            completion = self.backend.encode_len(content)
            return ChatResult(
                content=content,
                usage=Usage(
                    prompt_tokens=n,
                    completion_tokens=completion,
                    total_tokens=n + completion,
                ),
                model=model,
            )
        return ChatResult(
            content=content,
            usage=Usage(prompt_tokens=n, completion_tokens=2, total_tokens=n + 2),
            model=model,
        )


def test_detect_recovers_qwen_like_tokenizer():
    reg = load_registry()
    if not reg.available_families():
        return
    fake = FakeClient("qwen")
    result = run_detect(
        reg,
        fake,
        DetectConfig(model="anon/model", max_calls=12, use_completion=True),
    )
    assert result.calls_used >= 5
    top = result.verdict.candidates[0]
    assert top.family_id in ("qwen", "mimo")
    assert result.verdict.verdict != "unknown"
    # equivalence should group qwen/mimo
    assert "qwen" in result.verdict.verdict or "mimo" in result.verdict.verdict


def test_detect_recovers_deepseek_like_tokenizer():
    reg = load_registry()
    if not reg.available_families():
        return
    fake = FakeClient("deepseek")
    result = run_detect(
        reg,
        fake,
        DetectConfig(model="anon/model", max_calls=12, use_completion=True),
    )
    top = result.verdict.candidates[0]
    assert top.family_id in ("deepseek", "step")
    assert result.verdict.verdict not in ("unknown", "qwen+mimo")


def test_detect_recovers_minimax():
    reg = load_registry()
    if not reg.available_families():
        return
    fake = FakeClient("minimax")
    result = run_detect(
        reg,
        fake,
        DetectConfig(model="anon/model", max_calls=12, use_completion=True),
    )
    top = result.verdict.candidates[0]
    assert top.family_id == "minimax"
    assert result.verdict.verdict == "minimax"

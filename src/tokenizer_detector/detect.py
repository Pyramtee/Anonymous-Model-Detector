"""Detection pipeline: run probes against OpenRouter and score known families."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .client import OpenRouterClient
from .fuse import SignalScores, Verdict, fuse, group_equivalent_families
from .probes import BOUNDARY_PROBES, ECHO_PROBES, LONG_PROBES, SPECIAL_PROBES
from .registry import Family, Registry
from .tokenizer_backends import TokenizerBackend, load_backend
from .signals.completion_count import EchoSample, score_completion
from .signals.prompt_count import CountSample, score_prompt_counts
from .signals.text_stats import TextStatSample, score_text_stats


@dataclass
class DetectConfig:
    model: str
    max_calls: int = 12
    use_completion: bool = True
    rel_tol: float = 0.02
    abs_tol: float = 12.0
    unknown_threshold: float = 0.55
    temperature: float = 0.0


@dataclass
class DetectResult:
    verdict: Verdict
    calls_used: int
    raw: dict[str, Any] = field(default_factory=dict)


def _local_count(tokenizer: TokenizerBackend, text: str) -> int:
    return tokenizer.encode_len(text)


def _load_family_tokenizers(
    available: list[Family], tokenizer_dir: Path | None = None
) -> dict[str, TokenizerBackend]:
    from .registry import DEFAULT_TOKENIZER_DIR

    root = tokenizer_dir or DEFAULT_TOKENIZER_DIR
    toks: dict[str, TokenizerBackend] = {}
    for fam in available:
        if fam.tokenizer_path is None:
            continue
        try:
            toks[fam.id] = load_backend(fam.id, root)
        except Exception:
            # fall back to direct path if layout differs
            if fam.tokenizer_path.exists() and fam.tokenizer_path.name == "tokenizer.json":
                toks[fam.id] = load_backend(fam.id, fam.tokenizer_path.parent.parent)
    return toks


def run_detect(
    registry: Registry,
    client: OpenRouterClient,
    config: DetectConfig,
    progress: Callable[[str], None] | None = None,
) -> DetectResult:
    log = progress or (lambda _msg: None)
    available = registry.available_families()
    if not available:
        raise RuntimeError(
            "No local tokenizers found. Run `tokenizer-detector fetch` first."
        )

    tokenizers = _load_family_tokenizers(available)
    family_ids = list(tokenizers.keys())
    if not family_ids:
        raise RuntimeError(
            "Local tokenizers failed to load. Run `tokenizer-detector fetch` again."
        )
    meta = {
        f.id: {
            "vendor": f.vendor,
            "display": f.display,
            "models": f.models,
        }
        for f in available
    }

    calls = 0

    # --- Capability probe (short) ---
    log("capability probe (1 call)")
    cap = client.chat_with_nonce(
        config.model, "ping", max_tokens=8, temperature=config.temperature
    )
    calls += 1
    if cap.usage.is_zero_or_missing:
        raise RuntimeError(
            "API returned zero/missing usage. Detection requires prompt_tokens "
            "from OpenRouter usage. Cache hits zero usage — retried automatically; "
            "if this persists the endpoint may not report usage."
        )

    # --- S1: long discriminative probes ---
    count_samples: list[CountSample] = []
    long_probes = LONG_PROBES
    # Prefer special probes early — they break BPE-sharing ties cheaply
    special_budget = min(len(SPECIAL_PROBES), max(6, config.max_calls // 3))
    boundary_budget = min(3, max(1, config.max_calls // 5))
    echo_budget = min(3, max(0, (config.max_calls - 1 - len(long_probes) - boundary_budget - special_budget) if config.use_completion else 0))

    def _run_count_probe(probe) -> None:
        nonlocal calls
        log(f"probe {probe.id} ({calls + 1} calls)")
        result = client.chat_with_nonce(
            config.model,
            probe.text,
            max_tokens=4,
            temperature=config.temperature,
        )
        calls += 1
        api_n = result.usage.prompt_tokens or 0
        local = {fid: _local_count(tok, probe.text) for fid, tok in tokenizers.items()}
        count_samples.append(
            CountSample(
                probe_id=probe.id,
                api_prompt_tokens=api_n,
                local_counts=local,
                api_prompt_len_chars=len(probe.text),
            )
        )

    # Long probes first (delta signal), then special, then boundary
    for probe in long_probes:
        if calls >= config.max_calls - boundary_budget - special_budget - (echo_budget if config.use_completion else 0):
            break
        _run_count_probe(probe)

    for probe in SPECIAL_PROBES[:special_budget]:
        if calls >= config.max_calls - boundary_budget - (echo_budget if config.use_completion else 0):
            break
        _run_count_probe(probe)

    # --- Boundary probes (relative patterns) ---
    bnd_probes = BOUNDARY_PROBES[:boundary_budget]
    for probe in bnd_probes:
        if calls >= config.max_calls - (echo_budget if config.use_completion else 0):
            break
        _run_count_probe(probe)

    # --- Echo probes for S2/S3 ---
    echo_samples: list[EchoSample] = []
    text_stat_samples: list[TextStatSample] = []
    if config.use_completion and echo_budget > 0:
        for probe in ECHO_PROBES[:echo_budget]:
            if calls >= config.max_calls:
                break
            log(f"echo probe {probe.id} ({calls + 1} calls)")
            instruction = (
                "Please copy the following text exactly, character for character, "
                "with no additions and no explanations. Output only the text.\n\n"
                f"{probe.text}"
            )
            result = client.chat_with_nonce(
                config.model,
                instruction,
                max_tokens=max(64, len(probe.text) * 2),
                temperature=0.0,
            )
            calls += 1
            local = {
                fid: _local_count(tok, probe.text) for fid, tok in tokenizers.items()
            }
            echo_samples.append(
                EchoSample(
                    probe_id=probe.id,
                    expected_text=probe.text,
                    model_output=result.content,
                    api_completion_tokens=result.usage.completion_tokens,
                    local_counts=local,
                )
            )
            # Text stats use model output tokenized locally vs API completion count
            local_out = {
                fid: _local_count(tok, result.content) for fid, tok in tokenizers.items()
            }
            text_stat_samples.append(
                TextStatSample(
                    probe_id=probe.id,
                    text=result.content,
                    api_completion_tokens=result.usage.completion_tokens,
                    local_counts=local_out,
                )
            )

    # Drop samples with missing/zero usage so they don't poison deltas
    count_samples = [s for s in count_samples if s.api_prompt_tokens and s.api_prompt_tokens > 0]
    if len(count_samples) < 2:
        raise RuntimeError(
            "Not enough usable usage counts. The endpoint may omit usage, "
            "or responses were cache-zeroed / rate-limited."
        )

    log("scoring")
    s1 = {r.family_id: r.score for r in score_prompt_counts(
        count_samples, family_ids, rel_tol=config.rel_tol, abs_tol=config.abs_tol
    )}
    s2 = {r.family_id: r.score for r in score_completion(echo_samples, family_ids)}
    s3 = {r.family_id: r.score for r in score_text_stats(text_stat_samples, family_ids)}

    # Equivalence classes from local probe fingerprints (incl. special tokens)
    fingerprint = {
        fid: tuple(
            tok.encode_len(p.text) for p in (LONG_PROBES + BOUNDARY_PROBES + SPECIAL_PROBES)
        )
        for fid, tok in tokenizers.items()
    }
    equivalence = group_equivalent_families(family_ids, fingerprint)

    signals = SignalScores(prompt_count=s1, completion=s2, text_stats=s3)
    # If no completion signals, reweight toward prompt_count
    weights = {"prompt_count": 0.55, "completion": 0.25, "text_stats": 0.20}
    if not echo_samples:
        weights = {"prompt_count": 0.75, "completion": 0.0, "text_stats": 0.25}
        # text_stats also empty then
        if not text_stat_samples:
            weights = {"prompt_count": 1.0, "completion": 0.0, "text_stats": 0.0}

    verdict = fuse(
        family_ids=family_ids,
        signals=signals,
        weights=weights,
        unknown_threshold=config.unknown_threshold,
        meta=meta,
        equivalence=equivalence,
    )

    raw = {
        "capability_usage": {
            "prompt_tokens": cap.usage.prompt_tokens,
            "completion_tokens": cap.usage.completion_tokens,
            "total_tokens": cap.usage.total_tokens,
        },
        "count_samples": [
            {
                "probe_id": s.probe_id,
                "api_prompt_tokens": s.api_prompt_tokens,
                "local_counts": s.local_counts,
            }
            for s in count_samples
        ],
        "echo_samples": [
            {
                "probe_id": s.probe_id,
                "api_completion_tokens": s.api_completion_tokens,
                "fidelity_ok": True,
                "output_preview": s.model_output[:80],
            }
            for s in echo_samples
        ],
        "signals": {
            "prompt_count": s1,
            "completion": s2,
            "text_stats": s3,
        },
        "weights_used": weights,
        "equivalence": equivalence,
    }
    return DetectResult(verdict=verdict, calls_used=calls, raw=raw)

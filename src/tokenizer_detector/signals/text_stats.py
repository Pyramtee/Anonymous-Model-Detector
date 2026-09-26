"""S3: statistical fingerprint from generated text (chars/token ratios etc.)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextStatSample:
    probe_id: str
    text: str
    api_completion_tokens: int | None
    local_counts: dict[str, int]  # family_id -> tokenize(text) length


@dataclass
class TextStatScore:
    family_id: str
    score: float
    ratio_api: float | None = None
    ratio_local: float | None = None


def _chars_over_tokens(text: str, n_tokens: int | None) -> float | None:
    if not n_tokens or n_tokens <= 0:
        return None
    # Count unicode code points as a stable proxy
    return len(text) / n_tokens


def score_text_stats(
    samples: list[TextStatSample],
    family_ids: list[str],
    rel_tol: float = 0.08,
) -> list[TextStatScore]:
    """Compare API chars/token ratio vs each family's local ratio.

    Weaker signal than count alignment, but useful when counts are ambiguous.
    """
    usable = [
        s
        for s in samples
        if s.api_completion_tokens and s.text and s.api_completion_tokens > 0
    ]
    if not usable:
        return [TextStatScore(family_id=fid, score=0.0) for fid in family_ids]

    results: list[TextStatScore] = []
    for fid in family_ids:
        pair_scores: list[float] = []
        last_api_ratio: float | None = None
        last_local_ratio: float | None = None
        for sample in usable:
            api_ratio = _chars_over_tokens(sample.text, sample.api_completion_tokens)
            local_n = sample.local_counts.get(fid)
            local_ratio = _chars_over_tokens(sample.text, local_n)
            if api_ratio is None or local_ratio is None:
                continue
            last_api_ratio = api_ratio
            last_local_ratio = local_ratio
            denom = local_ratio * rel_tol
            if denom <= 0:
                denom = 0.05
            err = abs(api_ratio - local_ratio)
            pair_scores.append(max(0.0, 1.0 - min(1.0, err / denom)))
        score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        results.append(
            TextStatScore(
                family_id=fid,
                score=score,
                ratio_api=last_api_ratio,
                ratio_local=last_local_ratio,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results

"""S2: completion-side re-tokenization of echoed text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EchoSample:
    probe_id: str
    expected_text: str
    model_output: str
    api_completion_tokens: int | None
    local_counts: dict[str, int]  # family_id -> tokenize(expected or output)


@dataclass
class CompletionScore:
    family_id: str
    score: float
    detail: str = ""


def _normalize_for_copy(text: str) -> str:
    return "".join(text.split())


def _copy_fidelity(expected: str, output: str) -> float:
    """How faithfully the model echoed the target (0-1)."""
    if not expected:
        return 0.0
    exp = _normalize_for_copy(expected)
    out = _normalize_for_copy(output)
    if not out:
        return 0.0
    if exp in out or out in exp:
        return 1.0
    # Character-level overlap ratio
    exp_set = set(exp)
    out_set = set(out)
    if not exp_set:
        return 0.0
    inter = len(exp_set & out_set)
    return inter / len(exp_set)


def _count_match(api_n: int | None, local_n: int, rel_tol: float = 0.03, abs_tol: float = 4.0) -> float:
    if api_n is None or api_n <= 0:
        return 0.0
    denom = local_n * rel_tol + abs_tol
    err = abs(api_n - local_n)
    return max(0.0, 1.0 - min(1.0, err / denom))


def score_completion(
    samples: list[EchoSample],
    family_ids: list[str],
) -> list[CompletionScore]:
    """Score families using completion token counts on echo probes.

    When the model faithfully copies text, completion_tokens should match
    local tokenization of that text (plus small decode overhead).
    """
    if not samples:
        return [
            CompletionScore(family_id=fid, score=0.0, detail="no echo samples")
            for fid in family_ids
        ]

    results: list[CompletionScore] = []
    for fid in family_ids:
        scores: list[float] = []
        for sample in samples:
            fidelity = _copy_fidelity(sample.expected_text, sample.model_output)
            local_n = sample.local_counts.get(fid)
            if local_n is None:
                continue
            # If model didn't copy, completion_tokens is not comparable — skip.
            if fidelity < 0.5:
                continue
            s = _count_match(sample.api_completion_tokens, local_n)
            # Downweight slightly if fidelity is imperfect
            scores.append(s * (0.7 + 0.3 * fidelity))
        if not scores:
            results.append(
                CompletionScore(family_id=fid, score=0.0, detail="no usable echo")
            )
        else:
            results.append(
                CompletionScore(
                    family_id=fid,
                    score=sum(scores) / len(scores),
                    detail=f"n={len(scores)}",
                )
            )
    results.sort(key=lambda r: r.score, reverse=True)
    return results

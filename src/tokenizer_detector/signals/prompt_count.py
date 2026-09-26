"""S1: prompt_tokens alignment against local tokenizer counts.

Uses cross-probe deltas so chat-template overhead cancels out:
    api[i] - api[j]  ≈  local[i] - local[j]
plus a mild prior that mean(api - local) is a small positive template offset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CountSample:
    probe_id: str
    api_prompt_tokens: int
    local_counts: dict[str, int]  # family_id -> token count
    api_prompt_len_chars: int


@dataclass
class CountScore:
    family_id: str
    score: float
    abs_errors: dict[str, float]  # probe_id -> |api - local|
    notes: str = ""


def _delta_score(errors: list[float]) -> float:
    """Convert mean absolute delta error to [0,1]."""
    if not errors:
        return 0.0
    mean_err = sum(errors) / len(errors)
    # ~0 error => 1.0; mean error of 6+ tokens on deltas is already poor
    return max(0.0, 1.0 - min(1.0, mean_err / 6.0))


def _residual_bonus(residuals: list[float]) -> float:
    """Bonus in [0, 1] for plausible, consistent template overhead."""
    if not residuals:
        return 0.0
    n = len(residuals)
    mean_r = sum(residuals) / n
    var = sum((r - mean_r) ** 2 for r in residuals) / n
    std = var ** 0.5

    # Prefer mean residual in [0, 40] (template overhead or exact match)
    if mean_r < -2:
        mean_fit = 0.0
    elif mean_r <= 25:
        mean_fit = 1.0
    elif mean_r <= 50:
        mean_fit = max(0.0, 1.0 - (mean_r - 25) / 25.0)
    else:
        mean_fit = 0.0

    # Prefer low residual variance (constant offset)
    std_fit = max(0.0, 1.0 - min(1.0, std / 8.0))
    return 0.5 * mean_fit + 0.5 * std_fit


def score_prompt_counts(
    samples: list[CountSample],
    family_ids: list[str],
    rel_tol: float = 0.02,
    abs_tol: float = 12.0,
) -> list[CountScore]:
    """Score each family primarily by cross-probe delta agreement.

    ``rel_tol`` / ``abs_tol`` are kept for API compatibility and used only in
    the residual prior's ideal band.
    """
    if not samples:
        return [
            CountScore(family_id=fid, score=0.0, abs_errors={}, notes="no samples")
            for fid in family_ids
        ]

    # Prefer longer probes for delta pairs (less relative noise), but always
    # include short/special probes so added-token ties still participate.
    ordered = sorted(samples, key=lambda s: s.api_prompt_len_chars, reverse=True)
    long_samples = [s for s in ordered if s.api_prompt_len_chars >= 80]
    short_samples = [s for s in ordered if s.api_prompt_len_chars < 80]
    # Use all long samples (capped) + all short/special samples
    delta_samples = (long_samples[:8] + short_samples)[:16]

    results: list[CountScore] = []
    for fid in family_ids:
        errors: dict[str, float] = {}
        residuals: list[float] = []
        api_vec: list[int] = []
        local_vec: list[int] = []
        for sample in samples:
            local_n = sample.local_counts.get(fid)
            if local_n is None:
                continue
            errors[sample.probe_id] = abs(sample.api_prompt_tokens - local_n)
            residuals.append(sample.api_prompt_tokens - local_n)

        for sample in delta_samples:
            local_n = sample.local_counts.get(fid)
            if local_n is None:
                continue
            api_vec.append(sample.api_prompt_tokens)
            local_vec.append(local_n)

        delta_errors: list[float] = []
        for i in range(len(api_vec)):
            for j in range(i + 1, len(api_vec)):
                d_api = api_vec[i] - api_vec[j]
                d_local = local_vec[i] - local_vec[j]
                delta_errors.append(abs(d_api - d_local))

        d_score = _delta_score(delta_errors)
        r_bonus = _residual_bonus(residuals)
        # Deltas dominate; residual prior is a tie-breaker.
        score = 0.75 * d_score + 0.25 * r_bonus

        results.append(
            CountScore(
                family_id=fid,
                score=score,
                abs_errors=errors,
                notes=f"delta={d_score:.2f},residual={r_bonus:.2f}",
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results

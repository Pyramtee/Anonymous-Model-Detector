"""Multi-signal fusion into vendor-family confidence scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


DEFAULT_WEIGHTS = {
    "prompt_count": 0.55,
    "completion": 0.25,
    "text_stats": 0.20,
}

DEFAULT_UNKNOWN_THRESHOLD = 0.55
DEFAULT_AMBIGUITY_MARGIN = 0.08


@dataclass
class SignalScores:
    prompt_count: dict[str, float] = field(default_factory=dict)
    completion: dict[str, float] = field(default_factory=dict)
    text_stats: dict[str, float] = field(default_factory=dict)


@dataclass
class Candidate:
    family_id: str
    confidence: float
    combined_score: float
    signals: dict[str, float]
    shared_models: list[str] = field(default_factory=list)
    vendor: str = ""
    display: str = ""
    equivalent_families: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    verdict: str  # family_id | family_id-equivalence | "unknown" | "ambiguous"
    candidates: list[Candidate]
    unknown_score: float
    top_confidence: float
    explanation: str = ""


def group_equivalent_families(
    family_ids: list[str],
    fingerprint: dict[str, tuple],
) -> dict[str, list[str]]:
    """Map family_id -> list of families with an identical fingerprint tuple."""
    groups: dict[tuple, list[str]] = {}
    for fid in family_ids:
        key = fingerprint.get(fid, (fid,))
        groups.setdefault(key, []).append(fid)
    out: dict[str, list[str]] = {}
    for members in groups.values():
        for fid in members:
            out[fid] = list(members)
    return out


def fuse(
    family_ids: list[str],
    signals: SignalScores,
    weights: dict[str, float] | None = None,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    meta: dict[str, dict] | None = None,
    equivalence: dict[str, list[str]] | None = None,
) -> Verdict:
    """Combine weighted signals into a softmax-like confidence distribution."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    meta = meta or {}

    # Normalize weights to sum to 1 among provided signals
    total_w = sum(weights.values())
    if total_w <= 0:
        weights = dict(DEFAULT_WEIGHTS)
        total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    raw: dict[str, float] = {}
    detail: dict[str, dict[str, float]] = {}
    for fid in family_ids:
        parts: dict[str, float] = {}
        s = 0.0
        for key, w in weights.items():
            signal_map = getattr(signals, key, None) or {}
            val = float(signal_map.get(fid, 0.0))
            parts[key] = val
            s += w * val
        raw[fid] = s
        detail[fid] = parts

    # Softmax over scores (scaled for a peakier distribution)
    if not raw:
        return Verdict(
            verdict="unknown",
            candidates=[],
            unknown_score=1.0,
            top_confidence=0.0,
            explanation="no families available",
        )

    scale = 8.0
    max_raw = max(raw.values())
    exps = {fid: math.exp((v - max_raw) * scale) for fid, v in raw.items()}
    z = sum(exps.values()) or 1.0
    confidences = {fid: exps[fid] / z for fid in raw}

    candidates = []
    for fid, conf in sorted(confidences.items(), key=lambda kv: kv[1], reverse=True):
        m = meta.get(fid, {})
        eq = list((equivalence or {}).get(fid) or [fid])
        eq = [e for e in eq if e != fid]
        candidates.append(
            Candidate(
                family_id=fid,
                confidence=conf,
                combined_score=raw[fid],
                signals=detail[fid],
                shared_models=list(m.get("models") or []),
                vendor=str(m.get("vendor") or ""),
                display=str(m.get("display") or fid),
                equivalent_families=eq,
            )
        )

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    second_conf = second.confidence if second else 0.0
    unknown_score = max(0.0, 1.0 - top.combined_score)
    # Map unknown_score also from low absolute confidence
    unknown_score = max(unknown_score, 1.0 - top.confidence)

    # Merge verdict label when top is indistinguishable from equivalents
    top_label = top.family_id
    if top.equivalent_families:
        top_label = "+".join([top.family_id, *top.equivalent_families])

    if top.combined_score < unknown_threshold and top.confidence < 0.55:
        verdict = "unknown"
        explanation = (
            f"top score {top.combined_score:.2f} below threshold {unknown_threshold:.2f}"
        )
    elif (
        second is not None
        and top.confidence - second_conf < ambiguity_margin
        and top.combined_score < 0.35
        and second.family_id not in top.equivalent_families
    ):
        verdict = "ambiguous"
        explanation = (
            f"top two confidences too close: {top.confidence:.2f} vs {second_conf:.2f}"
        )
    else:
        # Accept a clear relative winner even when absolute scores are moderate
        # (chat-template noise keeps absolute scores below 1.0 in production).
        verdict = top_label
        if top.equivalent_families:
            explanation = (
                f"best match {top_label} (indistinguishable tokenizer) "
                f"confidence {top.confidence:.2f}"
            )
        else:
            explanation = f"best match {top.family_id} ({top.confidence:.2f})"

    return Verdict(
        verdict=verdict,
        candidates=candidates,
        unknown_score=unknown_score,
        top_confidence=top.confidence,
        explanation=explanation,
    )

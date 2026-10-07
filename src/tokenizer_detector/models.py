"""Shared, JSON-serializable contracts for the offline detector and runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 2
ALGORITHM_VERSION = "strict-v2.0"


@dataclass(frozen=True)
class Probe:
    probe_id: str
    payload: str
    text: str
    category: str
    split: str  # anchor, selection, confirmation, output
    bundle_id: str | None = None
    protocol: str = "ordinary"
    round: int = 0


@dataclass(frozen=True)
class Profile:
    profile_id: str
    reference_id: str
    family_id: str
    vendor: str
    display: str
    policy: str
    counts: dict[str, int]


@dataclass
class ProbeBank:
    probes: dict[str, Probe]
    profiles: dict[str, Profile]
    identity: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeBank:
        return cls(
            probes={k: Probe(**v) for k, v in data["probes"].items()},
            profiles={k: Profile(**v) for k, v in data["profiles"].items()},
            identity=data["identity"], diagnostics=data.get("diagnostics", {}),
        )


@dataclass(frozen=True)
class Observation:
    attempt_id: str
    probe_id: str
    group_id: str
    phase: str  # anchor, selection, confirmation, output, output_confirmation
    prompt_tokens: int | None
    provider: str | None = None
    actual_model: str | None = None
    valid: bool = True
    reason: str | None = None
    # Optional reliable output vocabulary evidence. None means no evidence.
    allowed_profiles: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Action:
    kind: str  # send, freeze, stop
    probe_ids: tuple[str, ...] = ()
    phase: str = "selection"
    reason: str | None = None


@dataclass
class Conclusion:
    status: str
    candidates: list[dict[str, Any]]
    confirmation: dict[str, Any]
    stop_reason: str
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

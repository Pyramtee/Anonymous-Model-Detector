"""Pure, replayable inference over finite tokenizer hypotheses.

Every public transition returns a new state.  The bank is immutable by convention;
``to_dict`` / ``from_dict`` include it so a saved state needs no network access.
The runner, not this module, owns HTTP attempts and passes the remaining budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .models import Action, Conclusion, Observation, Probe, ProbeBank


@dataclass
class InferenceState:
    bank: ProbeBank
    mode: str = "quick"
    observations: list[Observation] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    markers_required: bool = False
    fresh_anchor_required: bool = False
    continuation_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceState:
        values = dict(data)
        values["bank"] = ProbeBank.from_dict(values["bank"])
        values["observations"] = [
            Observation(**{**o, "allowed_profiles": (
                tuple(o["allowed_profiles"]) if o.get("allowed_profiles") is not None else None
            )}) for o in values.get("observations", [])
        ]
        return cls(**values)


def new_state(bank: ProbeBank, mode: str = "quick") -> InferenceState:
    if mode not in {"quick", "deep"}:
        raise ValueError("mode must be quick or deep")
    for profile in bank.profiles.values():
        if any(type(profile.counts.get(pid)) is not int or profile.counts[pid] < 0
               for pid in bank.probes):
            raise ValueError(f"incomplete count matrix for {profile.profile_id}")
    return InferenceState(bank=bank, mode=mode)


def _usable(o: Observation) -> bool:
    return o.valid and type(o.prompt_tokens) is int and o.prompt_tokens > 0


def _training(state: InferenceState) -> list[Observation]:
    return [o for o in state.observations if _usable(o) and (
        o.phase in {"anchor", "selection"} or o.attempt_id in state.promoted
    )]


def _fits(state: InferenceState) -> dict[str, dict[str, int]]:
    fits: dict[str, dict[str, int]] = {}
    for pid, profile in state.bank.profiles.items():
        offsets: dict[str, int] = {}
        for obs in _training(state):
            residual = obs.prompt_tokens - profile.counts[obs.probe_id]
            if obs.group_id in offsets and offsets[obs.group_id] != residual:
                break
            offsets[obs.group_id] = residual
        else:
            fits[pid] = offsets
    return fits


def _round_observations(state: InferenceState, rnd: dict) -> list[Observation]:
    phase = "output_confirmation" if rnd["source"] == "output" else "confirmation"
    return [o for o in state.observations[rnd["start_index"]:rnd.get("end_index")]
            if o.phase == phase and o.probe_id in rnd["probe_ids"]]


def _survivors(state: InferenceState, *, include_output: bool = True) -> dict[str, dict[str, int]]:
    fits = _fits(state)
    for rnd in state.rounds:
        if rnd["source"] == "output" and not include_output:
            continue
        fits = {p: b for p, b in fits.items() if p in rnd["fits"]}
        for obs in _round_observations(state, rnd):
            if obs.group_id != rnd["group_id"] or not obs.valid:
                continue
            if rnd["source"] == "output":
                if obs.allowed_profiles is not None:
                    fits = {p: b for p, b in fits.items() if p in obs.allowed_profiles}
            elif _usable(obs):
                fits = {p: b for p, b in fits.items()
                        if obs.prompt_tokens == state.bank.profiles[p].counts[obs.probe_id]
                        + rnd["fits"][p][obs.group_id]}
    if include_output:
        for obs in state.observations:
            if obs.phase == "output" and obs.valid and obs.allowed_profiles is not None:
                fits = {p: b for p, b in fits.items() if p in obs.allowed_profiles}
    return fits


def _families(state: InferenceState, profiles) -> set[str]:
    return {state.bank.profiles[p].family_id for p in profiles}


def _anchor_id(state: InferenceState) -> str | None:
    return next((p.probe_id for p in sorted(state.bank.probes.values(), key=lambda p: p.probe_id)
                 if p.split == "anchor"), None)


def _input_observations(state: InferenceState) -> list[Observation]:
    return [o for o in state.observations if _usable(o)
            and o.phase in {"anchor", "selection", "confirmation"}]


def _unstable(state: InferenceState) -> bool:
    observations = _input_observations(state)
    # A new route/configuration is not silently joined to the previous run.
    if len({o.group_id for o in observations}) > 1:
        return True
    seen: dict[tuple[str, str], int] = {}
    for obs in observations:
        key = (obs.group_id, obs.probe_id)
        if key in seen and seen[key] != obs.prompt_tokens:
            return True
        seen[key] = obs.prompt_tokens
    return False


def _group(state: InferenceState) -> str | None:
    return next((o.group_id for o in reversed(_training(state))), None)


def _selected(state: InferenceState) -> set[str]:
    return {o.probe_id for o in _training(state) if o.phase != "anchor"}


def _selection_text_count(state: InferenceState) -> int:
    return len({state.bank.probes[pid].text for pid in _selected(state)})


def _anchors(state: InferenceState) -> list[int]:
    """Indices of anchors separated by at least one different usable input."""
    result: list[int] = []
    anchor = _anchor_id(state)
    for i, obs in enumerate(state.observations):
        if obs.phase != "anchor" or obs.probe_id != anchor or not _usable(obs):
            continue
        if not result or any(_usable(o) and o.phase == "selection"
                             and state.bank.probes[o.probe_id].text != state.bank.probes[anchor].text
                             for o in state.observations[result[-1] + 1:i]):
            result.append(i)
    return result


def _has_input_prerequisites(state: InferenceState) -> bool:
    return len(_anchors(state)) >= 3 and _selection_text_count(state) >= 3


def observe(state: InferenceState, observation: Observation) -> InferenceState:
    if observation.probe_id not in state.bank.probes:
        raise ValueError(f"unknown probe {observation.probe_id}")
    if any(o.attempt_id == observation.attempt_id for o in state.observations):
        raise ValueError(f"duplicate attempt {observation.attempt_id}")
    rounds = state.rounds
    if observation.phase == "output" and rounds and rounds[-1]["source"] == "input":
        rounds = [dict(r) for r in rounds]
        rounds[-1].setdefault("end_index", len(state.observations))
    updated = replace(state, observations=[*state.observations, observation], rounds=rounds)
    if observation.phase == "anchor" and _usable(observation):
        updated.fresh_anchor_required = False
    families = _families(updated, _survivors(updated, include_output=False))
    if _selection_text_count(updated) >= 3 and families and families <= {"deepseek", "step"}:
        updated.markers_required = True
    return updated


def _direction(state: InferenceState, probes: list[Probe]) -> set[int]:
    """Directions are structural matrix facts, not labels inferred from names."""
    signs: set[int] = set()
    for probe in probes:
        values = {family: [p.counts[probe.probe_id] for p in state.bank.profiles.values()
                           if p.family_id == family] for family in ("step", "deepseek")}
        if values["step"] and values["deepseek"]:
            delta = min(values["step"]) - min(values["deepseek"])
            if delta:
                signs.add(1 if delta > 0 else -1)
    return signs


def _marker_directions(state: InferenceState) -> set[int]:
    selected = _selected(state)
    bundles: dict[str, list[Probe]] = {}
    for p in state.bank.probes.values():
        if p.split == "selection" and p.protocol == "markers":
            bundles.setdefault(p.bundle_id or p.probe_id, []).append(p)
    return set().union(*(_direction(state, ps) for ps in bundles.values()
                         if all(p.probe_id in selected for p in ps))) if bundles else set()


def _confirmation_plan(state: InferenceState, bank: ProbeBank, source: str) -> list[Probe]:
    used = {pid for r in state.rounds for pid in r["probe_ids"]}
    used.update(o.probe_id for o in state.observations
                if o.phase in {"confirmation", "output", "output_confirmation"})
    if source == "output":
        return sorted((p for p in bank.probes.values() if p.split == "output"
                       and p.probe_id not in used), key=lambda p: (p.round, p.probe_id))[:2]
    protocol = "markers" if state.markers_required else "ordinary"
    for number in sorted({p.round for p in bank.probes.values()
                          if p.split == "confirmation" and p.protocol == protocol}):
        probes = sorted((p for p in bank.probes.values() if p.split == "confirmation"
                         and p.protocol == protocol and p.round == number), key=lambda p: p.probe_id)
        if len(probes) >= 2 and not any(p.probe_id in used for p in probes):
            return probes[:2]
    return []


def _informative(state: InferenceState, probes: list[Probe]) -> bool:
    anchor = _anchor_id(state)
    if anchor is None or len(probes) < 2:
        return False
    survivors = _survivors(state, include_output=False)
    families = _families(state, survivors)
    profiles = list(state.bank.profiles.values())
    if not all(any(a.family_id != b.family_id and
                   a.counts[p.probe_id] - a.counts[anchor]
                   != b.counts[p.probe_id] - b.counts[anchor]
                   for i, a in enumerate(profiles) for b in profiles[i + 1:])
               for p in probes):
        return False
    def different(a, b):
        return any(a.counts[p.probe_id] - a.counts[anchor]
                   != b.counts[p.probe_id] - b.counts[anchor] for p in probes)
    # A confirmed unique result must independently distinguish every excluded
    # family. Multiple policies in that family cannot be quietly discarded.
    if len(families) == 1:
        return bool(survivors) and all(
            all(different(state.bank.profiles[pid], other) for other in profiles
                if other.family_id not in families) for pid in survivors
        ) and any(p.family_id not in families for p in profiles)
    # An ambiguous group can be confirmed even when its members are equivalent
    # on this channel, but the holdout must test some cross-family contrast.
    return any(a.family_id != b.family_id and different(a, b)
               for i, a in enumerate(profiles) for b in profiles[i + 1:])


def _output_started(state: InferenceState) -> bool:
    return any(o.phase == "output" for o in state.observations)


def freeze_confirmation(state: InferenceState, bank: ProbeBank) -> InferenceState:
    source = "output" if _output_started(state) else "input"
    probes = _confirmation_plan(state, bank, source)
    if len(probes) != 2:
        raise ValueError("two unused confirmation probes are required")
    if not _has_input_prerequisites(state) or _unstable(state):
        raise ValueError("input prerequisites are not satisfied")
    if source == "input" and state.markers_required and _marker_directions(state) != {-1, 1}:
        raise ValueError("both marker directions must be measured before confirmation")
    group = (_group(state) if source == "input" else
             next(o.group_id for o in reversed(state.observations) if o.phase == "output"))
    rounds = [dict(r) for r in state.rounds]
    if rounds:
        rounds[-1].setdefault("end_index", len(state.observations))
    rounds.append({"source": source, "group_id": group,
                   "probe_ids": [p.probe_id for p in probes],
                   "fits": _survivors(state), "start_index": len(state.observations),
                   "input_fits": _survivors(state, include_output=False),
                   "informative": source == "output" or _informative(state, probes)})
    return replace(state, rounds=rounds)


def _round_complete(state: InferenceState, rnd: dict) -> bool:
    observations = [o for o in _round_observations(state, rnd)
                    if o.group_id == rnd["group_id"] and o.valid and (
                        o.allowed_profiles is not None if rnd["source"] == "output" else _usable(o))]
    good = {o.probe_id for o in observations}
    if len(good.intersection(rnd["probe_ids"])) != 2 or not rnd["informative"]:
        return False
    if rnd["source"] == "output":
        # Reproduce the decisive family exclusions using the two fresh
        # holdouts alone. The exploratory output cannot do their work for them.
        independent = set(rnd["input_fits"])
        for obs in observations:
            independent.intersection_update(obs.allowed_profiles)
        return _families(state, independent) <= _families(state, _survivors(state))
    return True


def _selection_choices(state: InferenceState) -> list[list[Probe]]:
    used = {o.probe_id for o in state.observations if o.phase == "selection"}
    groups: dict[str, list[Probe]] = {}
    for p in state.bank.probes.values():
        if p.split == "selection":
            groups.setdefault(p.bundle_id or p.probe_id, []).append(p)
    pending = []
    for ps in groups.values():
        measured = {p.probe_id for p in ps}.intersection(_selected(state))
        if measured and len(measured) < len(ps):
            # A partial bundle can only continue with its other member.
            pending.append(sorted((p for p in ps if p.probe_id not in measured), key=lambda p: p.probe_id))
    if pending:
        return pending[:1]
    ordinary = _selection_text_count(state) < 3
    missing_directions = {-1, 1} - _marker_directions(state)
    choices = []
    for ps in groups.values():
        if any(p.probe_id in used for p in ps):
            continue
        marker = any(p.protocol == "markers" for p in ps)
        if ordinary and marker:
            continue
        if state.markers_required and missing_directions:
            if not marker or not (_direction(state, ps) & missing_directions):
                continue
        choices.append(sorted(ps, key=lambda p: p.probe_id))
    return choices


def _rank_choice(state: InferenceState, probes: list[Probe]) -> tuple:
    survivors = _survivors(state)
    group = _group(state)
    outcomes: dict[tuple[int, ...], set[str]] = {}
    profiles: dict[tuple[int, ...], set[str]] = {}
    for pid, offsets in survivors.items():
        if group not in offsets:
            continue
        prediction = tuple(state.bank.profiles[pid].counts[p.probe_id] + offsets[group] for p in probes)
        outcomes.setdefault(prediction, set()).add(state.bank.profiles[pid].family_id)
        profiles.setdefault(prediction, set()).add(pid)
    vectors = sorted(outcomes)
    separation = min((sum(abs(x - y) for x, y in zip(a, b))
                      for i, a in enumerate(vectors) for b in vectors[i + 1:]), default=0)
    return (max(map(len, outcomes.values()), default=0),
            max(map(len, profiles.values()), default=0),
            -separation, len(probes), sum(len(p.text) for p in probes),
            tuple(p.probe_id for p in probes))


def _can_split(state: InferenceState, probes: list[Probe]) -> bool:
    group = _group(state)
    predictions = {tuple(state.bank.profiles[pid].counts[p.probe_id] + offsets[group]
                         for p in probes)
                   for pid, offsets in _survivors(state).items() if group in offsets}
    return len(predictions) > 1


def next_action(state: InferenceState, bank: ProbeBank, remaining: int) -> Action:
    stop = lambda reason: Action("stop", reason=reason)
    if not bank.profiles or _anchor_id(state) is None:
        return stop("reference_unavailable")
    if _unstable(state):
        return stop("unstable_endpoint")
    active = state.rounds[-1] if state.rounds and "end_index" not in state.rounds[-1] else None
    if active and _round_complete(state, active) and not state.fresh_anchor_required:
        return stop("confirmed")
    if remaining <= 0:
        return stop("budget_exhausted")
    if state.fresh_anchor_required:
        return Action("send", (_anchor_id(state),), "anchor")
    if active:
        observed = _round_observations(state, active)
        done = {o.probe_id for o in observed}
        todo = [p for p in active["probe_ids"] if p not in done]
        if not todo:
            return stop("confirmed" if _round_complete(state, active) else "confirmation_unusable")
        if remaining < len(todo):
            return stop("confirmation_budget_insufficient")
        return Action("send", (todo[0],),
                      "output_confirmation" if active["source"] == "output" else "confirmation")
    if _output_started(state):
        if not any(o.phase == "output" and o.valid and o.allowed_profiles is not None
                   for o in state.observations):
            return stop("output_evidence_unavailable")
        return Action("freeze") if remaining >= 2 and len(_confirmation_plan(state, bank, "output")) == 2 else stop("confirmation_budget_insufficient")
    anchors = _anchors(state)
    if not anchors:
        return Action("send", (_anchor_id(state),), "anchor")
    since_anchor = any(_usable(o) and o.phase == "selection" for o in state.observations[anchors[-1] + 1:])
    if len(anchors) == 1 and _selection_text_count(state) >= 2 and since_anchor:
        return Action("send", (_anchor_id(state),), "anchor")
    survivors = _survivors(state)
    marker_done = not state.markers_required or _marker_directions(state) == {-1, 1}
    reserve = max(0, 3 - len(anchors)) + 2
    choices = [ps for ps in _selection_choices(state) if len(ps) + reserve <= remaining]
    enough_selection = _selection_text_count(state) >= 3
    if enough_selection and marker_done:
        # Equal predictions cannot shrink the set. Stop the input experiment
        # after the mandatory protocol rather than spending the output budget.
        choices = [ps for ps in choices if _can_split(state, ps)]
    finish_selection = enough_selection and marker_done and (
        len(_families(state, survivors)) <= 1 or not choices or remaining <= reserve)
    if finish_selection:
        if len(anchors) < 3:
            if since_anchor:
                return Action("send", (_anchor_id(state),), "anchor")
            return stop("insufficient_anchor_spacing")
        if not survivors:
            return stop("no_reference_match")
        plan = _confirmation_plan(state, bank, "input")
        if len(plan) < 2 or not _informative(state, plan):
            return stop("confirmation_protocol_unavailable")
        return Action("freeze") if remaining >= 2 else stop("confirmation_budget_insufficient")
    if choices:
        probes = min(choices, key=lambda ps: _rank_choice(state, ps))
        return Action("send", tuple(p.probe_id for p in probes), "selection")
    return stop("insufficient_evidence" if not enough_selection or not marker_done else "probe_pool_exhausted")


def continue_state(state: InferenceState, mode: str = "deep") -> InferenceState:
    if mode not in {"quick", "deep"}:
        raise ValueError("mode must be quick or deep")
    rounds = [dict(r) for r in state.rounds]
    if rounds:
        last = rounds[-1]
        resume_output = (last["source"] == "output" and "end_index" not in last
                         and not _round_complete(state, last) and bool(_survivors(state))
                         and {o.probe_id for o in _round_observations(state, last)}
                         != set(last["probe_ids"])
                         and all(o.valid and o.allowed_profiles is not None
                                 and o.group_id == last["group_id"]
                                 for o in _round_observations(state, last)))
        if resume_output:
            # The already-frozen output hypothesis is unchanged. A fresh input
            # anchor must pass before the outstanding holdout is sent.
            last["continued_at"] = len(state.observations)
        else:
            last.setdefault("end_index", len(state.observations))
    promoted = sorted(set(state.promoted) | {o.attempt_id for o in state.observations
                                            if o.phase == "confirmation" and _usable(o)})
    return replace(state, mode=mode, rounds=rounds, promoted=promoted,
                   fresh_anchor_required=True, continuation_index=len(state.observations))


def _hypothesis_evidence(state: InferenceState, surviving: dict) -> list[dict[str, Any]]:
    """Include rejected profiles and the exact observations that rejected them."""
    records = []
    training_ids = {o.attempt_id for o in _training(state)}
    for pid, profile in sorted(state.bank.profiles.items()):
        offsets: dict[str, int] = {}
        evidence = []
        for obs in state.observations:
            row: dict[str, Any] = {"attempt_id": obs.attempt_id, "probe_id": obs.probe_id,
                                   "phase": obs.phase, "group_id": obs.group_id}
            if obs.attempt_id in training_ids:
                residual = obs.prompt_tokens - profile.counts[obs.probe_id]
                offset = offsets.setdefault(obs.group_id, residual)
                row.update(local_count=profile.counts[obs.probe_id], observed=obs.prompt_tokens,
                           offset=offset, residual=residual - offset,
                           accepted=residual == offset, basis="input_fit")
            elif obs.phase == "confirmation" and _usable(obs):
                rnd = next((r for r in state.rounds if r["source"] == "input"
                            and obs in _round_observations(state, r)), None)
                if rnd is None or pid not in rnd["fits"] or obs.group_id != rnd["group_id"]:
                    continue
                offset = rnd["fits"][pid][obs.group_id]
                residual = obs.prompt_tokens - profile.counts[obs.probe_id] - offset
                row.update(local_count=profile.counts[obs.probe_id], observed=obs.prompt_tokens,
                           offset=offset, residual=residual, accepted=residual == 0,
                           basis="frozen_input_confirmation")
            elif obs.phase in {"output", "output_confirmation"} and obs.valid and obs.allowed_profiles is not None:
                if obs.phase == "output_confirmation" and not any(
                        r["source"] == "output" and obs.group_id == r["group_id"]
                        and obs in _round_observations(state, r) for r in state.rounds):
                    continue
                row.update(accepted=pid in obs.allowed_profiles, basis="output_vocabulary")
            else:
                continue
            evidence.append(row)
        records.append({"profile_id": pid, "family": profile.family_id,
                        "reference_id": profile.reference_id, "policy": profile.policy,
                        "survives": pid in surviving, "offsets": offsets,
                        "evidence": evidence,
                        "exclusions": [r for r in evidence if not r["accepted"]]})
    return records


def conclude(state: InferenceState, stop_reason: str | None = None) -> Conclusion:
    fits = _survivors(state)
    families = _families(state, fits)
    pending_output = (_output_started(state) and
                      (not state.rounds or state.rounds[-1]["source"] != "output") and
                      any(o.phase == "output" and o.valid and o.allowed_profiles is not None
                          for o in state.observations))
    complete = bool(state.rounds and _round_complete(state, state.rounds[-1])
                    and (state.rounds[-1]["start_index"] >= state.continuation_index
                         or state.rounds[-1].get("continued_at") == state.continuation_index)
                    and not pending_output)
    output_complete = bool(complete and state.rounds and state.rounds[-1]["source"] == "output")
    confirmed_source = state.rounds[-1]["source"] if state.rounds else None
    reason = stop_reason or ("confirmed" if complete else "insufficient_evidence")
    status = "insufficient_evidence"
    if not state.bank.profiles:
        status = "reference_unavailable"
    elif _unstable(state) or reason in {"unstable_endpoint", "output_route_changed"}:
        status = "unstable_endpoint"
    elif (not fits and _selection_text_count(state) >= 3 and len(_anchors(state)) >= 2
          and (not _survivors(state, include_output=False) or complete)):
        status = "no_reference_match"
    elif complete and _has_input_prerequisites(state) and not state.fresh_anchor_required:
        status = "compatible" if len(families) == 1 else "ambiguous"
    elif (len(families) > 1 and families == _families(state, _survivors(state, include_output=False))
          and _has_input_prerequisites(state) and not state.fresh_anchor_required
          and any(r["source"] == "input" and _round_complete(state, r) for r in state.rounds)):
        # An inconclusive output channel does not erase a completed input-only
        # ambiguity when it has supplied no new family exclusion.
        status, complete, confirmed_source = "ambiguous", True, "input"
    elif reason in {"request_failed", "invalid_response", "outcome_unknown",
                    "retry_after_exceeds_wait_limit"}:
        status = "request_failed"
    candidates = []
    for family in sorted(families):
        profiles = sorted(p for p in fits if state.bank.profiles[p].family_id == family)
        reference = state.bank.profiles[profiles[0]]
        candidates.append({"family": family, "vendor": reference.vendor,
                           "display": reference.display, "profiles": profiles,
                           "references": sorted({state.bank.profiles[p].reference_id for p in profiles})})
    return Conclusion(status=status, candidates=candidates,
                      confirmation={"complete": complete, "rounds": len(state.rounds),
                                    "source": confirmed_source, "output_complete": output_complete},
                      stop_reason=reason,
                      hypotheses=_hypothesis_evidence(state, fits),
                      diagnostics={"selection_texts": _selection_text_count(state),
                                   "spaced_anchors": len(_anchors(state)),
                                   "markers_required": state.markers_required,
                                   "observations": len(state.observations)})

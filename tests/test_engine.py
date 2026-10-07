"""Synthetic oracle matrices: no production tokenizer or network dependencies."""
import json
from dataclasses import replace

import pytest

from tokenizer_detector import engine
from tokenizer_detector.models import Observation, Probe, ProbeBank, Profile


def bank(ambiguous=False, markers=False):
    probes = {"anchor": Probe("anchor", "a", "a", "anchor", "anchor")}
    for i in range(3):
        pid = f"s{i}"
        probes[pid] = Probe(pid, pid, pid, "text", "selection")
    for r in range(3):
        for i in range(2):
            pid = f"c{r}{i}"
            probes[pid] = Probe(pid, pid, pid, "text", "confirmation", round=r)
            if markers:
                pid = f"mc{r}{i}"
                probes[pid] = Probe(pid, pid, pid, "marker", "confirmation",
                                    protocol="markers", round=r)
    for i in range(3):
        pid = f"o{i}"
        probes[pid] = Probe(pid, pid, pid, "output", "output")
    if markers:
        for direction in ("ds", "step"):
            for part in ("raw", "control"):
                pid = f"{direction}_{part}"
                probes[pid] = Probe(pid, pid, pid, "marker", "selection",
                                    bundle_id=direction, protocol="markers")
    base = {pid: i + 10 for i, pid in enumerate(probes)}
    profiles = {}
    families = ["alpha", "beta", "gamma"] if ambiguous else ["alpha", "beta"]
    if markers:
        families = ["deepseek", "step"]
    for i, family in enumerate(families):
        for policy in (["native", "plain"] if markers else ["plain"]):
            pid = f"{family}:{policy}"
            counts = dict(base)
            if not markers and (i > 1 or i == 1 and not ambiguous):
                counts.update({p: n * 2 - base["anchor"] for p, n in base.items()})
            if markers:
                for direction in ("ds", "step"):
                    counts[f"{direction}_raw"] = 10 if policy == "native" and (
                        family == "deepseek" and direction == "ds" or
                        family == "step" and direction == "step") else 20
                    counts[f"{direction}_control"] = 25
                for r in range(3):
                    counts[f"mc{r}0"] = 10 if family == "deepseek" and policy == "native" else 20
                    counts[f"mc{r}1"] = 10 if family == "step" and policy == "native" else 20
            profiles[pid] = Profile(pid, family, family, family, family, policy, counts)
    return ProbeBank(probes, profiles, {"synthetic": True})


def add(state, pid, phase, profile="alpha:plain", *, count=None, group="g", allowed=None, valid=True):
    if count is None:
        count = state.bank.profiles[profile].counts[pid] + 37
    return engine.observe(state, Observation(str(len(state.observations)), pid, group,
                                            phase, count, valid=valid,
                                            allowed_profiles=allowed))


def run(state, profile="alpha:plain", budget=12):
    trace = []
    while len(trace) < 100:
        action = engine.next_action(state, state.bank, budget - len(state.observations))
        trace.append(action)
        if action.kind == "stop":
            return state, trace
        if action.kind == "freeze":
            state = engine.freeze_confirmation(state, state.bank)
        else:
            assert len(action.probe_ids) <= budget - len(state.observations)
            for pid in action.probe_ids:
                state = add(state, pid, action.phase, profile)
    pytest.fail("state machine did not terminate")


def until_freeze(state, profile="alpha:plain", budget=12):
    for _ in range(30):
        action = engine.next_action(state, state.bank, budget - len(state.observations))
        if action.kind == "freeze":
            return engine.freeze_confirmation(state, state.bank)
        assert action.kind == "send", action
        for pid in action.probe_ids:
            state = add(state, pid, action.phase, profile)
    pytest.fail("did not freeze")


def test_unique_requires_three_spaced_anchors_and_two_frozen_holdouts():
    state, trace = run(engine.new_state(bank()))
    result = engine.conclude(state)
    assert result.status == "compatible"
    assert result.candidates[0]["family"] == "alpha"
    assert result.diagnostics["spaced_anchors"] == 3
    assert result.diagnostics["selection_texts"] == 3
    assert len(state.observations) == 8
    assert [o.phase for o in state.observations] == [
        "anchor", "selection", "selection", "anchor", "selection", "anchor",
        "confirmation", "confirmation"]
    assert trace[-1].reason == "confirmed"
    assert engine.next_action(state, state.bank, 0).reason == "confirmed"


@pytest.mark.parametrize("family", ["deepseek", "step"])
def test_marker_native_both_directions_and_controls_fit_exactly_twelve(family):
    state, trace = run(engine.new_state(bank(markers=True)), f"{family}:native")
    assert engine.conclude(state).status == "compatible"
    assert engine.conclude(state).candidates[0]["family"] == family
    assert len(state.observations) == 12
    bundles = [a for a in trace if len(a.probe_ids) == 2]
    assert len(bundles) == 2
    assert all({p.rsplit("_", 1)[1] for p in a.probe_ids} == {"raw", "control"}
               for a in bundles)
    assert state.rounds[0]["probe_ids"] == ["mc00", "mc01"]
    assert trace[-1].reason == "confirmed"


def test_shared_plain_marker_policies_remain_ambiguous():
    state, _ = run(engine.new_state(bank(markers=True)), "step:plain")
    result = engine.conclude(state)
    assert result.status == "ambiguous"
    assert {c["family"] for c in result.candidates} == {"deepseek", "step"}
    assert all(c["profiles"] == [c["family"] + ":plain"] for c in result.candidates)


def test_insufficient_budget_preserves_candidates_without_unconfirmed_verdict():
    state, _ = run(engine.new_state(bank(markers=True)), "step:native", budget=10)
    result = engine.conclude(state)
    assert result.status == "insufficient_evidence"
    assert len(state.observations) <= 10
    assert not result.confirmation["complete"]


def test_holdouts_are_not_used_to_refit_offset_or_accept_single_confirmation():
    state = until_freeze(engine.new_state(bank()))
    frozen = state.rounds[0]["fits"]
    first, second = state.rounds[0]["probe_ids"]
    state = add(state, first, "confirmation")
    assert engine.conclude(state).status == "insufficient_evidence"
    state = add(state, second, "confirmation", count=999)
    assert state.rounds[0]["fits"] == frozen
    assert engine.conclude(state).status == "no_reference_match"
    assert any(row["basis"] == "frozen_input_confirmation" and not row["accepted"]
               for h in engine.conclude(state).hypotheses for row in h["evidence"])


def test_two_consecutive_anchors_do_not_satisfy_spacing():
    state = engine.new_state(bank())
    for _ in range(3):
        state = add(state, "anchor", "anchor")
    for pid in ("s0", "s1", "s2"):
        state = add(state, pid, "selection")
    assert engine.conclude(state).diagnostics["spaced_anchors"] == 1
    with pytest.raises(ValueError, match="prerequisites"):
        engine.freeze_confirmation(state, state.bank)


@pytest.mark.parametrize("group,count", [("other", None), ("g", 99)])
def test_group_change_or_repeated_anchor_drift_is_unstable(group, count):
    state = add(engine.new_state(bank()), "anchor", "anchor")
    state = add(state, "anchor", "anchor", group=group, count=count)
    assert engine.next_action(state, state.bank, 10).reason == "unstable_endpoint"
    assert engine.conclude(state).status == "unstable_endpoint"


def test_all_hypotheses_and_exclusion_observations_are_reported():
    state, _ = run(engine.new_state(bank()))
    result = engine.conclude(state)
    assert len(result.hypotheses) == 2
    rejected = next(h for h in result.hypotheses if h["profile_id"] == "beta:plain")
    assert not rejected["survives"]
    assert rejected["exclusions"]
    assert all(r["residual"] != 0 for r in rejected["exclusions"])


def test_failure_reports_candidates_and_never_invents_confirmation():
    state = add(engine.new_state(bank()), "anchor", "anchor")
    result = engine.conclude(state, "request_failed")
    assert result.status == "request_failed"
    assert len(result.candidates) == 2


def test_uninformative_second_holdout_cannot_confirm():
    b = bank()
    beta = b.profiles["beta:plain"]
    counts = {**beta.counts, "c01": b.profiles["alpha:plain"].counts["c01"]}
    b.profiles["beta:plain"] = replace(beta, counts=counts)
    state, trace = run(engine.new_state(b))
    assert trace[-1].reason == "confirmation_protocol_unavailable"
    assert engine.conclude(state).status == "insufficient_evidence"


@pytest.mark.parametrize("mode", ["quick", "deep"])
def test_continue_demands_fresh_anchor_and_new_confirmation_round(mode):
    state, _ = run(engine.new_state(bank()))
    prior = state.to_dict()
    resumed = engine.continue_state(state, mode)
    assert state.to_dict() == prior
    assert engine.conclude(resumed).status == "insufficient_evidence"
    assert engine.next_action(resumed, resumed.bank, 32).phase == "anchor"
    resumed, _ = run(resumed, budget=40)
    assert engine.conclude(resumed).status == "compatible"
    assert len(resumed.observations) == 11
    assert resumed.rounds[-1]["probe_ids"] == ["c10", "c11"]


def test_output_counts_do_not_enter_input_fit_and_two_fresh_holds_are_required():
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    assert engine.conclude(state).status == "ambiguous"
    state = add(state, "o0", "output", count=9999, group="output-route",
                allowed=("alpha:plain",))
    assert engine.conclude(state).status == "insufficient_evidence"
    assert engine.next_action(state, state.bank, 10).kind == "freeze"
    state = engine.freeze_confirmation(state, state.bank)
    assert state.rounds[-1]["probe_ids"] == ["o1", "o2"]
    state = add(state, "o1", "output_confirmation", count=9998, group="output-route",
                allowed=("alpha:plain",))
    assert engine.conclude(state).status == "insufficient_evidence"
    state = add(state, "o2", "output_confirmation", count=9997, group="output-route",
                allowed=("alpha:plain",))
    result = engine.conclude(state)
    assert result.status == "compatible"
    assert result.candidates[0]["family"] == "alpha"
    survivor = next(h for h in result.hypotheses if h["survives"])
    assert survivor["offsets"] == {"g": 37}


def test_missing_output_evidence_retains_confirmed_input_ambiguity_and_stops():
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    state = add(state, "o0", "output", count=999, group="out", allowed=None)
    assert engine.next_action(state, state.bank, 10).reason == "output_evidence_unavailable"
    assert engine.conclude(state).status == "ambiguous"


def test_wrong_output_confirmation_group_cannot_confirm():
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    state = add(state, "o0", "output", group="out", allowed=("alpha:plain",))
    state = engine.freeze_confirmation(state, state.bank)
    for pid in ("o1", "o2"):
        state = add(state, pid, "output_confirmation", group="wrong", allowed=("alpha:plain",))
    assert engine.conclude(state).status == "insufficient_evidence"
    assert engine.next_action(state, state.bank, 10).reason == "confirmation_unusable"


def test_serialization_is_json_roundtrippable_and_transitions_do_not_mutate():
    state = until_freeze(engine.new_state(bank()))
    before = state.to_dict()
    restored = engine.InferenceState.from_dict(json.loads(json.dumps(before)))
    assert engine.next_action(state, state.bank, 10) == engine.next_action(restored, restored.bank, 10)
    assert engine.conclude(state).to_dict() == engine.conclude(restored).to_dict()
    advanced = add(state, state.rounds[0]["probe_ids"][0], "confirmation")
    assert state.to_dict() == before
    assert len(advanced.observations) == len(state.observations) + 1
    with pytest.raises(ValueError, match="duplicate"):
        engine.observe(advanced, advanced.observations[-1])


def test_ties_prefer_larger_prediction_separation_deterministically():
    state = add(engine.new_state(bank()), "anchor", "anchor")
    action = engine.next_action(state, state.bank, 11)
    assert action.probe_ids == ("s2",)


def test_probe_aliases_do_not_count_as_distinct_selection_texts():
    b = bank()
    b.probes["s1"] = replace(b.probes["s1"], text=b.probes["s0"].text)
    state, _ = run(engine.new_state(b))
    assert engine.conclude(state).diagnostics["selection_texts"] == 2
    assert engine.conclude(state).status == "insufficient_evidence"


def test_partial_bundle_is_completed_before_another_probe_can_be_selected():
    state = engine.new_state(bank(markers=True))
    for pid, phase in [("anchor", "anchor"), ("s0", "selection"),
                       ("s1", "selection"), ("anchor", "anchor"), ("s2", "selection")]:
        state = add(state, pid, phase, "step:native")
    action = engine.next_action(state, state.bank, 7)
    assert len(action.probe_ids) == 2
    state = add(state, action.probe_ids[0], "selection", "step:native")
    following = engine.next_action(state, state.bank, 6)
    assert following.probe_ids == (action.probe_ids[1],)


def test_surviving_policy_cannot_borrow_another_policys_informative_holdout():
    b = bank()
    alpha = b.profiles["alpha:plain"]
    counts = dict(alpha.counts)
    for pid in ("c00", "c01"):
        counts[pid] = b.profiles["beta:plain"].counts[pid]
    b.profiles["alpha:native"] = replace(alpha, profile_id="alpha:native", policy="native", counts=counts)
    state, trace = run(engine.new_state(b))
    assert trace[-1].reason == "confirmation_protocol_unavailable"
    assert engine.conclude(state).status == "insufficient_evidence"


def test_invalid_count_cannot_supply_an_anchor_or_offset():
    state = add(engine.new_state(bank()), "anchor", "anchor", count=0)
    result = engine.conclude(state)
    assert result.diagnostics["spaced_anchors"] == 0
    assert all(h["offsets"] == {} for h in result.hypotheses)
    assert engine.next_action(state, state.bank, 11).phase == "anchor"


def test_output_excluding_all_profiles_also_needs_two_fresh_confirmations():
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    state = add(state, "o0", "output", group="out", allowed=())
    assert engine.conclude(state).status == "insufficient_evidence"
    state = engine.freeze_confirmation(state, state.bank)
    state = add(state, "o1", "output_confirmation", group="out", allowed=())
    assert engine.conclude(state).status == "insufficient_evidence"
    state = add(state, "o2", "output_confirmation", group="out", allowed=())
    assert engine.conclude(state).status == "no_reference_match"


def test_deep_stops_uninformative_selection_pool_to_leave_output_budget():
    b = bank(markers=True)
    for i in range(30):
        pid = f"redundant{i}"
        b.probes[pid] = Probe(pid, pid, pid, "ordinary", "selection")
        for profile in b.profiles.values():
            profile.counts[pid] = 50 + i
    state, _ = run(engine.new_state(b, "deep"), "step:plain", budget=40)
    assert len(state.observations) == 12
    assert engine.conclude(state).status == "ambiguous"


def test_resume_incomplete_output_keeps_frozen_holdout_after_fresh_anchor():
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    state = add(state, "o0", "output", group="out", allowed=("alpha:plain",))
    state = engine.freeze_confirmation(state, state.bank)
    state = add(state, "o1", "output_confirmation", group="out", allowed=("alpha:plain",))
    frozen = state.rounds[-1]["fits"]
    state = engine.continue_state(state)
    assert engine.next_action(state, state.bank, 25).phase == "anchor"
    assert engine.conclude(state).status == "insufficient_evidence"
    state = add(state, "anchor", "anchor")
    assert state.rounds[-1]["fits"] == frozen
    action = engine.next_action(state, state.bank, 24)
    assert action.phase == "output_confirmation"
    assert action.probe_ids == ("o2",)
    state = add(state, "o2", "output_confirmation", group="out", allowed=("alpha:plain",))
    assert engine.conclude(state).status == "compatible"


def test_contradictory_output_round_cannot_be_reused_on_resume():
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    state = add(state, "o0", "output", group="out", allowed=("alpha:plain",))
    state = engine.freeze_confirmation(state, state.bank)
    state = add(state, "o1", "output_confirmation", group="out", allowed=())
    state = engine.continue_state(state)
    state = add(state, "anchor", "anchor")
    assert engine.next_action(state, state.bank, 25).kind == "stop"
    assert engine.conclude(state).status == "insufficient_evidence"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_observed_output_route_change_overrides_previously_confirmed_result(ambiguous):
    state, _ = run(engine.new_state(bank(ambiguous=ambiguous)))
    previous = engine.conclude(state)
    assert previous.status in {"compatible", "ambiguous"}
    # The runner records the response in the journal but rejects it before
    # inserting an incomparable observation into the inference state.
    result = engine.conclude(state, stop_reason="output_route_changed")
    assert result.status == "unstable_endpoint"
    assert result.candidates == previous.candidates
    assert result.stop_reason == "output_route_changed"


@pytest.mark.parametrize("holdouts,expected", [
    ((("alpha:plain", "beta:plain"), ("alpha:plain", "beta:plain")), "insufficient_evidence"),
    ((("alpha:plain", "beta:plain"), ("alpha:plain",)), "compatible"),
    ((("beta:plain",), ("beta:plain",)), "insufficient_evidence"),
    ((("alpha:plain",), ("beta:plain",)), "no_reference_match"),
])
def test_output_holdouts_must_jointly_reproduce_exclusions_without_first_sample(holdouts, expected):
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    state = add(state, "o0", "output", group="out", allowed=("alpha:plain",))
    state = engine.freeze_confirmation(state, state.bank)
    for pid, allowed in zip(("o1", "o2"), holdouts):
        state = add(state, pid, "output_confirmation", group="out", allowed=allowed)
    result = engine.conclude(state)
    assert result.status == expected
    assert result.confirmation["output_complete"] == (expected in {"compatible", "no_reference_match"})


def test_output_without_new_family_exclusion_preserves_input_ambiguity():
    state, _ = run(engine.new_state(bank(ambiguous=True)))
    state = add(state, "o0", "output", group="out", allowed=("alpha:plain", "beta:plain"))
    state = engine.freeze_confirmation(state, state.bank)
    state = add(state, "o1", "output_confirmation", group="out", allowed=("alpha:plain", "beta:plain"))
    result = engine.conclude(state, "request_failed")
    assert result.status == "ambiguous"
    assert result.confirmation["complete"]
    assert result.confirmation["source"] == "input"
    assert not result.confirmation["output_complete"]

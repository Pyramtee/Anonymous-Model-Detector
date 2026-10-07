"""End-to-end acceptance against frozen independent official-tokenizer observations.

The mock reads static prompt counts; it never tokenizes a request using the
implementation under test to generate its expected response.
"""
import json
from pathlib import Path
import httpx
import pytest

from tokenizer_detector.probe_bank import build_probe_bank
from tokenizer_detector.references import load_adapters, load_manifest, DEFAULT_ARTIFACT_DIR
from tokenizer_detector.runner import RunConfig, execute_run, replay_run
from tokenizer_detector.models import ProbeBank

FIXTURE = json.loads((Path(__file__).parent / "fixtures/observations_pinned.json").read_text())
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pinned_bank():
    manifest = load_manifest()
    adapters = load_adapters(manifest, DEFAULT_ARTIFACT_DIR, include_controls=True)
    bank = build_probe_bank(manifest, adapters)
    assert bank.identity["reference_manifest"] == manifest
    assert bank.identity["probe_sha256"] == FIXTURE["provenance"]["probe_sha256"]
    assert bank.identity["manifest_sha256"] == FIXTURE["provenance"]["manifest_sha256"]
    return bank


@pytest.mark.parametrize("profile_id", sorted(FIXTURE["profiles"]))
def test_pinned_profile_through_http_runner_and_replay(profile_id, pinned_bank, tmp_path):
    bank = pinned_bank
    oracle = FIXTURE["profiles"][profile_id]
    by_text = {p.text: p.probe_id for p in bank.probes.values()}
    calls = []
    def handler(request):
        assert request.method == "POST"
        body = json.loads(request.content)
        pid = by_text[body["messages"][0]["content"]]
        calls.append(pid)
        return httpx.Response(200, json={"id": f"offline-{len(calls)}",
            "model": "anonymous", "provider": "offline-fixture",
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": oracle["prompt_counts"][pid], "completion_tokens": 0}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = execute_run(tmp_path, bank, RunConfig("anonymous", min_interval=0),
                             "fake-offline-key", client=client)
    assert len(calls) <= 12
    assert report == replay_run(tmp_path)
    if oracle["role"] == "external_control":
        assert report["status"] == "no_reference_match", report
    elif oracle["family_id"] in {"deepseek", "step"} and oracle["policy"] == "plain":
        assert report["status"] == "ambiguous", report
        assert {x["family"] for x in report["candidates"]} == {"deepseek", "step"}
    else:
        assert report["status"] == "compatible", report
        assert [x["family"] for x in report["candidates"]] == [oracle["family_id"]]
        assert report["confirmation"]["complete"]


@pytest.mark.parametrize("held_out", ["qwen", "mimo", "deepseek", "step", "kimi", "glm", "minimax", "hy"])
def test_leave_one_family_out_never_reintroduces_omitted_reference(held_out, pinned_bank, tmp_path):
    # Deliberately incomplete banks are for this offline stress experiment only.
    # The CLI continues to require every target in the pinned manifest.
    bank = ProbeBank(pinned_bank.probes,
                     {k: v for k, v in pinned_bank.profiles.items() if v.family_id != held_out},
                     {**pinned_bank.identity, "offline_held_out_family": held_out})
    by_text = {p.text: p.probe_id for p in bank.probes.values()}
    for pid, oracle in FIXTURE["profiles"].items():
        if oracle["family_id"] != held_out:
            continue
        def handler(request):
            body = json.loads(request.content)
            probe = by_text[body["messages"][0]["content"]]
            return httpx.Response(200, json={"model": "offline-heldout", "provider": "fixture",
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": oracle["prompt_counts"][probe], "completion_tokens": 0}})
        directory = tmp_path / pid.replace(":", "--")
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            report = execute_run(directory, bank, RunConfig("offline-heldout", min_interval=0),
                                 "offline-only", client=client)
        assert report == replay_run(directory)
        assert held_out not in {c["family"] for c in report["candidates"]}
        assert report["budget"]["used"] <= 12
        if held_out not in {"step", "deepseek"}:
            assert report["status"] == "no_reference_match", report
        else:
            # These families share the plain pipeline. With one omitted the
            # required two-direction marker protocol cannot be certified;
            # abstention is preferable to inventing an unsupported identity.
            assert report["status"] in {"insufficient_evidence", "no_reference_match"}, report

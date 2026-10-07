"""Offline integration tests REQUIRE previously fetched pinned artifacts."""
import json
from pathlib import Path
import unicodedata

import pytest

from tokenizer_detector.references import load_manifest, load_adapters, inspect_references
from tokenizer_detector.probes import LONG_PROBES
from tokenizer_detector.probe_bank import build_probe_bank

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def loaded():
    manifest = load_manifest()
    # Missing files fail explicitly; this suite never downloads or silently skips.
    return manifest, load_adapters(manifest, include_controls=True)


def test_all_ten_locked_references_are_ready(loaded):
    manifest, adapters = loaded
    assert len(adapters) == 10
    assert all(r['status'] == 'ready' for r in inspect_references(manifest))
    assert sum(r['role'] == 'target' for r in manifest['references']) == 8


def test_published_bare_long_counts(loaded):
    expected = {
        'qwen3_8': [314,347,312,303], 'mimo2_6': [341,347,300,279],
        'deepseek_v4_1': [316,334,292,289], 'step5': [316,334,292,289],
        'kimi_k3': [307,327,281,300], 'glm5_3': [318,329,280,293],
        'minimax_m3': [297,323,275,280], 'hy4': [301,328,283,285],
        'mistral_v0_3': [546,403,367,380], 'gpt_neox': [666,352,338,393],
    }
    for rid, adapter in loaded[1].items():
        assert [len(adapter.encode(p.text, 'literal_special').ids) for p in LONG_PROBES] == expected[rid]


def test_static_independent_golden_ids(loaded):
    fixture = json.loads((Path(__file__).parent / 'fixtures/reference_goldens.json').read_text())
    manifest, adapters = loaded
    for ref in manifest['references']:
        locked = fixture['provenance']['references'][ref['reference_id']]['artifact_sha256']
        assert locked == {a['repo_path']: a['sha256'] for a in ref['artifacts']}
    for case in fixture['cases']:
        actual = adapters[case['reference_id']].encode(case['text'], case['policy'])
        assert list(actual.ids) == case['ids'], (case['reference_id'], case['text'], case['policy'])


def test_special_markers_and_plain_ambiguity(loaded):
    adapters = loaded[1]
    step, ds = adapters['step5'], adapters['deepseek_v4_1']
    for marker, counts in [('<im_start>', (1,4)), ('<patch_start>', (1,4)),
                           ('<｜place▁holder▁no▁0｜>', (14,1))]:
        assert (len(step.encode(marker, 'literal_special').ids), len(ds.encode(marker, 'literal_special').ids)) == counts
        assert step.encode(marker, 'plain').ids == ds.encode(marker, 'plain').ids
    kimi = adapters['kimi_k3']
    assert kimi.encode('[BOS]', 'literal_special').ids == (163584,)
    assert kimi.encode('<|reserved_token_163592|>', 'literal_special').ids == (163592,)
    assert len(kimi.encode('[BOS]', 'plain').ids) > 1


def test_bytelevel_bytes_preserve_utf8_fragments(loaded):
    for rid in ('qwen3_8', 'mimo2_6', 'deepseek_v4_1', 'step5', 'kimi_k3'):
        adapter = loaded[1][rid]
        parts = adapter.encode('Hello 世界 🎉', 'plain').token_bytes
        assert parts is not None
        assert b''.join(parts) == 'Hello 世界 🎉'.encode()
        vocab = adapter.vocabulary_bytes()
        assert vocab is not None and b'\xe4' in vocab


def test_matrix_target_only_split_and_special_groups(loaded, tmp_path):
    bank = build_probe_bank(*loaded, cache_dir=tmp_path)
    assert len(bank.profiles) == 16
    assert len(bank.diagnostics['component_summaries']) == 8
    assert len(bank.diagnostics['component_differences']) == 28
    ds_step = next(d for d in bank.diagnostics['component_differences']
                   if set(d['references']) == {'deepseek_v4_1', 'step5'})
    assert 'vocab_hash' not in ds_step['different_components']
    assert 'merges_hash' not in ds_step['different_components']
    assert 'added_tokens_hash' in ds_step['different_components']
    assert {p.family_id for p in bank.profiles.values()} == {'qwen','mimo','deepseek','step','kimi','glm','minimax','hy'}
    assert bank.to_dict() == build_probe_bank(*loaded, cache_dir=tmp_path).to_dict()
    for protocol in ('ordinary', 'markers'):
        for round in range(3):
            rows = [p for p in bank.probes.values() if p.split == 'confirmation' and p.protocol == protocol and p.round == round]
            assert len(rows) == 2
    for p in bank.probes.values():
        if p.bundle_id:
            assert len([q for q in bank.probes.values() if q.bundle_id == p.bundle_id]) == 2
    assert bank.profiles['deepseek_v4_1:plain'].counts == bank.profiles['step5:plain'].counts
    assert bank.profiles['deepseek_v4_1:literal_special'].counts != bank.profiles['step5:literal_special'].counts
    assert all(p.split == 'selection' for w in bank.diagnostics['profile_pair_witnesses']
               if 'probe_id' in w for p in [bank.probes[w['probe_id']]])


def test_frozen_independent_observation_matrix_matches_current_probes(loaded):
    from tokenizer_detector.references import canonical_hash
    frozen = json.loads((Path(__file__).parent / 'fixtures/observations_pinned.json').read_text())
    import hashlib
    generator = Path(__file__).resolve().parents[1] / frozen['provenance']['source_script']
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == frozen['provenance']['source_script_sha256']
    bank = build_probe_bank(*loaded)
    assert frozen['provenance']['manifest_sha256'] == canonical_hash(loaded[0])
    assert frozen['provenance']['probe_sha256'] == bank.identity['probe_sha256']
    for pid, profile in bank.profiles.items():
        assert profile.counts == frozen['profiles'][pid]['counts']
        assert frozen['profiles'][pid]['prompt_counts'] == {k: n + 17 for k, n in profile.counts.items()}
    for probe in bank.probes.values():
        assert probe.text == 'Data:\n' + probe.payload + '\nEnd.'
    assert len(frozen['profiles']) == 20

import hashlib
import json

import httpx
import pytest
from tokenizers import Tokenizer, models

from tokenizer_detector import references


def tiny_manifest(tmp_path):
    raw = Tokenizer(models.BPE(vocab={'a': 0}, merges=[])).to_str().encode()
    artifact = {'repo_id': 'test/model', 'revision': 'a' * 40, 'repo_path': 'tokenizer.json',
                'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    ref = {'reference_id': 'tiny', 'family_id': 'tiny', 'display': 'Tiny', 'vendor': 'Test',
           'role': 'target', 'backend': 'hf_json_v1', 'policies': ['plain', 'literal_special'],
           'artifacts': [artifact]}
    return {'schema_version': 1, 'references': [ref]}, artifact, raw


def test_manifest_rejects_floating_revision(tmp_path):
    manifest, item, _ = tiny_manifest(tmp_path)
    item['revision'] = 'main'
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(references.ReferenceError, match='fixed commit'):
        references.load_manifest(path)


def test_missing_or_modified_artifact_never_silently_drops_family(tmp_path):
    manifest, item, raw = tiny_manifest(tmp_path)
    with pytest.raises(references.ReferenceError, match='tiny'):
        references.load_adapters(manifest, tmp_path)
    path = references.artifact_path(item, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    assert references.inspect_references(manifest, tmp_path)[0]['status'] == 'ready'
    path.write_bytes(b'x' * len(raw))
    assert references.inspect_references(manifest, tmp_path)[0]['artifacts'][0]['status'] == 'hash_mismatch'
    with pytest.raises(references.ReferenceError):
        references.load_adapters(manifest, tmp_path)


def test_fetch_checks_digest_before_replacing_and_pins_commit(tmp_path, monkeypatch):
    manifest, item, raw = tiny_manifest(tmp_path)
    calls = []
    real_client = httpx.Client
    def respond(req):
        calls.append(str(req.url))
        return httpx.Response(200, content=b'x' * len(raw))
    monkeypatch.setattr(references.httpx, 'Client', lambda **kwargs: real_client(transport=httpx.MockTransport(respond)))
    ready, errors = references.fetch_references(manifest, tmp_path)
    assert not ready and len(errors) == 1
    assert '/resolve/' + 'a' * 40 + '/tokenizer.json' in calls[0]
    assert not references.artifact_path(item, tmp_path).exists()
    assert not list(tmp_path.rglob('*.download'))


def test_fetch_rejects_unknown_family_without_http(tmp_path):
    manifest, _, _ = tiny_manifest(tmp_path)
    with pytest.raises(references.ReferenceError, match='Unknown selected'):
        references.fetch_references(manifest, tmp_path, families=['unknown'])


def test_matrix_cache_invalidates_inputs_and_rebuilds_corruption(tmp_path):
    from tokenizer_detector.adapters import Encoding
    from tokenizer_detector.probe_bank import build_probe_bank
    class FakeAdapter:
        implementation_version = 'test-1'
        supported_policies = ('plain',)
        calls = 0
        def encode(self, text, policy):
            self.calls += 1
            return Encoding(tuple(range(len(text))), None)
    manifest, _, _ = tiny_manifest(tmp_path)
    adapter = FakeAdapter()
    first = build_probe_bank(manifest, {'tiny': adapter}, cache_dir=tmp_path)
    calls = adapter.calls
    assert build_probe_bank(manifest, {'tiny': adapter}, cache_dir=tmp_path).to_dict() == first.to_dict()
    assert adapter.calls == calls
    cache = tmp_path / (first.identity['cache_key'] + '.json')
    cache.write_text('{incomplete')
    build_probe_bank(manifest, {'tiny': adapter}, cache_dir=tmp_path)
    assert adapter.calls > calls
    adapter.implementation_version = 'test-2'
    changed = build_probe_bank(manifest, {'tiny': adapter}, cache_dir=tmp_path)
    assert changed.identity['cache_key'] != first.identity['cache_key']
    manifest['references'][0]['artifacts'][0]['sha256'] = 'f' * 64
    changed_again = build_probe_bank(manifest, {'tiny': adapter}, cache_dir=tmp_path)
    assert changed_again.identity['cache_key'] != changed.identity['cache_key']

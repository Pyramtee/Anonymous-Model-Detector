import json

import pytest

from tokenizer_detector.run_store import RunLockedError, RunStore, RunStoreError, read_events


def test_create_resume_and_orphan_attempt_still_consumes_budget(tmp_path):
    path = tmp_path / "run"
    with RunStore(path, create=True) as store:
        payload = {"nested": [1]}
        store.append("run_started", config=payload)
        payload["nested"].append(2)
        store.append("attempt_started", attempt_id="orphan")
        assert store.events[0]["config"] == {"nested": [1]}
        assert store.attempts_used == 1
    with RunStore(path) as resumed:
        assert resumed.attempts_used == 1
        resumed.append("resume")
    assert [e["seq"] for e in read_events(path)] == [1, 2, 3]


def test_exclusive_lock_and_create_never_overwrites(tmp_path):
    with RunStore(tmp_path, create=True) as first:
        first.append("run_started")
        with pytest.raises(RunLockedError):
            with RunStore(tmp_path):
                pass
    original = (tmp_path / "events.jsonl").read_bytes()
    with pytest.raises(RunStoreError, match="already has"):
        with RunStore(tmp_path, create=True):
            pass
    assert (tmp_path / "events.jsonl").read_bytes() == original
    with RunStore(tmp_path):
        pass  # failed acquire/create must have released their own handles


@pytest.mark.parametrize("raw", [
    b'{"type":"run","seq":1,"timestamp":"now"}',
    b'{"type":"run","seq":2,"timestamp":"now"}\n',
    b'{"type":"run","seq":true,"timestamp":"now"}\n',
    b'{"type":"run","seq":1,"timestamp":"now","n":NaN}\n',
    b'{"type":"attempt_finished","seq":1,"timestamp":"now","attempt_id":"x"}\n',
    b'\xff\n', b'\n',
])
def test_corrupt_journal_is_rejected_without_repair(tmp_path, raw):
    path = tmp_path / "events.jsonl"
    path.write_bytes(raw)
    with pytest.raises(RunStoreError):
        read_events(tmp_path)
    with pytest.raises(RunStoreError):
        with RunStore(tmp_path):
            pass
    assert path.read_bytes() == raw


def test_fsync_failure_poison_store(tmp_path, monkeypatch):
    with RunStore(tmp_path, create=True) as store:
        def fail(_):
            raise OSError("disk failure")
        monkeypatch.setattr("tokenizer_detector.run_store.os.fsync", fail)
        with pytest.raises(RunStoreError, match="persist"):
            store.append("attempt_started", attempt_id="reserved")
        assert store.events == []
        with pytest.raises(RunStoreError, match="failed"):
            store.append("run_started")
    # A write whose durability was uncertain remains counted if recovered.
    assert read_events(tmp_path)[0]["attempt_id"] == "reserved"


def test_attempt_pairs_are_validated_before_writing(tmp_path):
    with RunStore(tmp_path, create=True) as store:
        with pytest.raises(RunStoreError):
            store.append("attempt_finished", attempt_id="missing")
        store.append("attempt_started", attempt_id="one")
        with pytest.raises(RunStoreError):
            store.append("attempt_started", attempt_id="one")
        store.append("attempt_finished", attempt_id="one")
        with pytest.raises(RunStoreError):
            store.append("attempt_finished", attempt_id="one")
    assert len(read_events(tmp_path)) == 2

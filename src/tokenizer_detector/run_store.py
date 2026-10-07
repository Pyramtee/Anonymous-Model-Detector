"""Durable, single-writer experiment journal. Reports are derived from this log."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any


class RunStoreError(RuntimeError):
    pass


class RunLockedError(RunStoreError):
    pass


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read a journal without acquiring a writer lock; never repair corrupt data."""
    path = Path(path)
    if path.is_dir():
        path = path / "events.jsonl"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RunStoreError(f"Cannot read journal {path}: {exc}") from exc
    if raw and not raw.endswith(b"\n"):
        raise RunStoreError("Truncated journal: final event has no newline")
    events: list[dict[str, Any]] = []
    started: set[str] = set()
    finished: set[str] = set()
    for index, line in enumerate(raw.splitlines(), 1):
        try:
            event = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
            if type(event.get("seq")) is not int or event["seq"] != index:
                raise ValueError("non-contiguous event sequence")
            if not isinstance(event.get("type"), str) or not event["type"]:
                raise ValueError("missing event type")
            if not isinstance(event.get("timestamp"), str):
                raise ValueError("missing timestamp")
            if event["type"] in {"attempt_started", "attempt_finished"}:
                aid = event.get("attempt_id")
                if not isinstance(aid, str) or not aid:
                    raise ValueError("missing attempt_id")
                if event["type"] == "attempt_started":
                    if aid in started:
                        raise ValueError("duplicate attempt_started")
                    started.add(aid)
                else:
                    if aid not in started or aid in finished:
                        raise ValueError("unmatched or duplicate attempt_finished")
                    finished.add(aid)
            events.append(event)
        except (UnicodeError, ValueError, TypeError) as exc:
            raise RunStoreError(f"Invalid journal event at line {index}: {exc}") from exc
    return events


class RunStore:
    """Hold the run lock for the entire online operation (Linux / WSL)."""

    def __init__(self, path: str | Path, create: bool = False) -> None:
        self.path = Path(path)
        self.create = create
        self.events: list[dict[str, Any]] = []
        self._lock = None
        self._journal = None
        self._failed = False

    @property
    def attempts_used(self) -> int:
        return sum(e["type"] == "attempt_started" for e in self.events)

    def __enter__(self) -> "RunStore":
        if self._lock is not None:
            raise RunStoreError("RunStore is already open")
        if self.create:
            self.path.mkdir(parents=True, exist_ok=True)
        if not self.path.is_dir():
            raise RunStoreError(f"Run directory does not exist: {self.path}")
        self._lock = (self.path / "run.lock").open("a+b")
        try:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunLockedError(f"Run is already in use: {self.path}") from exc
            journal = self.path / "events.jsonl"
            if self.create:
                try:
                    self._journal = journal.open("xb", buffering=0)
                except FileExistsError as exc:
                    raise RunStoreError(f"Run already has a journal: {self.path}") from exc
                os.fsync(self._journal.fileno())
                directory_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self.events = []
            else:
                self.events = read_events(journal)
                self._journal = journal.open("ab", buffering=0)
            self._failed = False
            return self
        except BaseException:
            self.close()
            raise

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        if self._journal is None or self._failed:
            raise RunStoreError("Journal is closed or has failed; refusing further events")
        if not isinstance(event_type, str) or not event_type:
            raise RunStoreError("Event type must be a nonempty string")
        if {"type", "seq", "timestamp"}.intersection(fields):
            raise RunStoreError("Reserved event fields cannot be overridden")
        if event_type in {"attempt_started", "attempt_finished"}:
            aid = fields.get("attempt_id")
            if not isinstance(aid, str) or not aid:
                raise RunStoreError("Attempt events require a nonempty attempt_id")
            related = [e["type"] for e in self.events if e.get("attempt_id") == aid]
            if event_type == "attempt_started" and "attempt_started" in related:
                raise RunStoreError("Duplicate attempt_started")
            if event_type == "attempt_finished" and ("attempt_started" not in related or "attempt_finished" in related):
                raise RunStoreError("Unmatched or duplicate attempt_finished")
        event = {
            "type": event_type,
            "seq": len(self.events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        try:
            encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            # Keep the replay state independent of caller-owned nested containers.
            event = json.loads(encoded)
            # Unbuffered files can write partially; finish the whole record before fsync.
            remaining = memoryview(encoded)
            while remaining:
                count = self._journal.write(remaining)
                if not count:
                    raise OSError("short journal write")
                remaining = remaining[count:]
            self._journal.flush()
            os.fsync(self._journal.fileno())
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            self._failed = True
            raise RunStoreError(f"Cannot persist journal event: {exc}") from exc
        self.events.append(event)
        return event

    def close(self) -> None:
        try:
            if self._journal is not None:
                self._journal.close()
        finally:
            self._journal = None
            if self._lock is not None:
                try:
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
                finally:
                    self._lock.close()
                    self._lock = None

    def __exit__(self, *args: object) -> None:
        self.close()

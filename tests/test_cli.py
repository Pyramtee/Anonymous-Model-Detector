import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from tokenizer_detector import cli
from tokenizer_detector.models import SCHEMA_VERSION
from tokenizer_detector.report import render_markdown, write_reports
from tokenizer_detector.runner import RunConfig, execute_run
from test_runner import manual_bank, response_for


def test_help_exposes_six_commands_without_loading_credentials(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("offline help loaded credentials"))
    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("fetch", "list", "fingerprint", "detect", "resume", "replay"):
        assert command in result.stdout


@pytest.mark.parametrize("flag,value", [("--max-calls", "12"), ("--unknown-threshold", "0.5"), ("--no-completion", None)])
def test_old_parameters_have_explicit_migration_message(flag, value):
    args = ["detect", "--model", "anonymous", flag] + ([value] if value else [])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert "--max-requests" in result.stdout


def test_missing_references_fail_before_key_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_key", lambda _: pytest.fail("missing references should stop before credentials"))
    output = tmp_path / "report.json"
    result = CliRunner().invoke(cli.app, ["detect", "--model", "anonymous", "--artifacts",
                                        str(tmp_path / "missing"), "-o", str(output)])
    assert result.exit_code == 2
    assert json.loads(output.read_text())["status"] == "reference_unavailable"


def test_replay_is_offline_keyless_and_does_not_overwrite_original(tmp_path, monkeypatch):
    bank = manual_bank()
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response_for(bank, r)))) as client:
        report = execute_run(tmp_path / "run", bank,
                             RunConfig("anonymous", min_interval=0), "fake-key", client=client)
    write_reports(tmp_path / "run", report)
    original = (tmp_path / "run/report.json").read_bytes()
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("replay read credentials"))
    monkeypatch.setattr(cli, "_references", lambda *_: pytest.fail("replay loaded current tokenizers"))
    output = tmp_path / "derived.json"
    result = CliRunner().invoke(cli.app, ["replay", str(tmp_path / "run"), "-o", str(output)])
    assert result.exit_code == 0, result.exception
    assert json.loads(output.read_text()) == report
    assert (tmp_path / "run/report.json").read_bytes() == original
    assert "归属概率" in result.stdout


def test_old_report_is_not_silently_migrated(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"verdict":"qwen","confidence":0.99}')
    result = CliRunner().invoke(cli.app, ["replay", str(path)])
    assert result.exit_code == 2


def test_reports_use_one_result_and_have_no_confidence_fields(tmp_path):
    payload = {"schema_version": SCHEMA_VERSION, "status": "ambiguous", "model": "匿名",
               "candidates": [{"family": "deepseek"}, {"family": "step"}],
               "confirmation": {"complete": True}, "stop_reason": "confirmed",
               "budget": {"used": 12, "limit": 12}}
    write_reports(tmp_path, payload)
    assert json.loads((tmp_path / "report.json").read_text()) == payload
    assert (tmp_path / "report.md").read_text() == render_markdown(payload)
    assert "deepseek" in render_markdown(payload) and "step" in render_markdown(payload)
    assert "confidence" not in json.dumps(payload)

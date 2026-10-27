"""Phase 2 command gates, without any real data."""

from pathlib import Path
import runpy
import sys

import pytest


@pytest.mark.parametrize("status,code", [("passed", 0), ("blocked", 2)])
def test_audit_command_respects_gate(monkeypatch, status, code):
    script = Path(__file__).resolve().parents[1] / "scripts/run_audit.py"
    command = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(command.__globals__, "run_audit", lambda cfg: {"status": status})
    monkeypatch.setattr(sys, "argv", ["run_audit.py"])
    assert command() == code


def test_split_command_runs_current_config(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/make_splits.py"
    command = runpy.run_path(str(script))["main"]
    captured = []
    monkeypatch.setitem(command.__globals__, "run_split_freeze", lambda cfg: captured.append(cfg))
    monkeypatch.setattr(sys, "argv", ["make_splits.py"])
    assert command() == 0
    assert captured[0]["split"] == {"train": .7, "val": .15, "test": .15}

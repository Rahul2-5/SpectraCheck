"""CLI wiring and low-sample gate, without downloading anything."""

from pathlib import Path
import runpy
import sys

import pytest


@pytest.mark.parametrize("ai_count,expected_status", [(1200, 0), (999, 2)])
def test_main_command_passes_config_and_gates_small_sample(monkeypatch, tmp_path, capsys,
                                                          ai_count, expected_status):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/download_main.py"))
    command = namespace["main"]
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return {"saved_counts": {"ai": ai_count, "real": 4000}}

    monkeypatch.setitem(command.__globals__, "stream_generator_subset", fake_download)
    monkeypatch.setattr(sys, "argv", ["download_main.py", "--out-dir", str(tmp_path / "main")])
    assert command() == expected_status
    args, kwargs = calls[0]
    assert args[:3] == ("TheKernel01/Tiny-GenImage", "SD15", ["train", "validation"])
    assert args[3] == tmp_path / "main"
    assert args[4:] == (4000, 42)
    assert kwargs["revision"] == "main"
    assert len(kwargs["configuration_hash"]) == 64
    assert ("STOP:" in capsys.readouterr().out) == (expected_status == 2)


def test_pilot_command_forwards_source_and_separate_output(monkeypatch, tmp_path):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/download_pilot.py"))
    command = namespace["main"]
    calls = []

    def fake_pilot(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(command.__globals__, "prepare_cifake_sample", fake_pilot)
    source, output = tmp_path / "cifake", tmp_path / "pilot"
    monkeypatch.setattr(sys, "argv", ["download_pilot.py", "--source-dir", str(source),
                                      "--out-dir", str(output)])
    assert command() == 0
    args, kwargs = calls[0]
    assert args == (source, output, 500, 42)
    assert kwargs["split_names"] == ["train", "test"]
    assert len(kwargs["configuration_hash"]) == 64

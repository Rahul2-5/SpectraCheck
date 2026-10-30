"""Synthetic tests for read-only geometry-bin diagnostics."""

from __future__ import annotations

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_geometry_bins.py"
SPEC = spec_from_file_location("diagnose_geometry_bins", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _image(size: tuple[int, int], fmt: str) -> bytes:
    stream = BytesIO()
    image = Image.new("RGB", size, (30, 90, 150))
    if fmt == "JPEG":
        image.save(stream, format=fmt, quality=90)
    else:
        image.save(stream, format=fmt)
    image.close()
    return stream.getvalue()


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    rows = []
    specs = (
        [(0, (64, 64), "JPEG") for _ in range(6)]
        + [(0, (96, 48), "JPEG") for _ in range(4)]
        + [(1, (64, 64), "PNG") for _ in range(2)]
        + [(1, (96, 48), "PNG") for _ in range(8)]
    )
    for index, (label, size, fmt) in enumerate(specs):
        identifier = f"row_{index:03d}"
        relative = Path("real" if label == 0 else "ai") / f"{identifier}.{fmt.lower()}"
        path = root / "main" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _image(size, fmt)
        path.write_bytes(payload)
        rows.append({
            "id": identifier,
            "rel_path": relative.as_posix(),
            "label": label,
            "generator": "Real" if label == 0 else "SD15",
            "source_split": "train",
            "ext": "jpg" if fmt == "JPEG" else "png",
            "n_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "config_hash": "fixture",
        })
    (root / "main").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "main" / "manifest.csv", index=False)
    (root / "main" / "acquisition.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    return root


def test_geometry_strategies_write_balanced_diagnostic_outputs(tmp_path):
    root = _data_root(tmp_path)
    output = tmp_path / "geometry-diagnostic"
    result = MODULE.run_geometry_analysis(
        root,
        output,
        scope="original",
        image_size=32,
        cv_folds=3,
    )

    assert result["splits_created"] is False
    assert not (root / "splits.csv").exists()
    summary = pd.read_csv(output / "geometry_bin_summary.csv")
    assert len(summary) == len(MODULE.STRATEGIES)
    assert summary["balanced"].all()
    assert (summary["selected_total"] > 0).all()

    probes = pd.read_csv(output / "geometry_bin_probe_results.csv")
    assert set(probes["probe_status"]) == {"ran"}
    assert set(probes["model"]) == {"logistic_regression", "decision_tree"}
    assert (probes["threshold"] == 0.60).all()
    assert (output / "geometry_bin_matching_summary.md").is_file()

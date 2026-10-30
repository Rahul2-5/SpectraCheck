"""Synthetic tests for read-only geometry-bin diagnostics."""

from __future__ import annotations

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
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


def test_distribution_rows_handles_empty_cohort_with_nan_quantiles():
    empty = pd.DataFrame({
        "id": pd.Series(dtype=str),
        "label": pd.Series(dtype=int),
        "width": pd.Series(dtype=float),
        "height": pd.Series(dtype=float),
        "aspect_ratio": pd.Series(dtype=float),
    })
    result = MODULE._distribution_rows(empty, "empty", "candidate")

    assert len(result) == 12
    assert set(result["count"]) == {0}
    assert set(result["numeric_count"]) == {0}
    assert set(result["status"]) == {"empty_cohort"}
    assert result["q01"].isna().all()
    assert result["median"].isna().all()


def test_distribution_rows_handles_empty_numeric_column():
    frame = pd.DataFrame({
        "id": ["real", "ai"],
        "label": [0, 1],
        "width": [64, 64],
        "height": [64, 64],
        "aspect_ratio": [np.nan, np.nan],
    })
    result = MODULE._distribution_rows(frame, "numeric_missing", "candidate")
    aspect = result[result["feature"] == "aspect_ratio"]

    assert set(aspect["count"]) == {1}
    assert set(aspect["numeric_count"]) == {0}
    assert set(aspect["missing_count"]) == {1}
    assert set(aspect["status"]) == {"empty_numeric"}
    assert aspect["q95"].isna().all()


def test_mixed_valid_and_empty_strategies_continue_and_record_status(tmp_path, monkeypatch):
    root = tmp_path / "data"
    rows = []
    specs = [(0, (64, 64), "JPEG") for _ in range(6)] + [(1, (65, 65), "PNG") for _ in range(6)]
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
    pd.DataFrame(rows).to_csv(root / "main" / "manifest.csv", index=False)
    (root / "main" / "acquisition.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    strategies = (
        MODULE.Strategy("empty", "fixed", resolution_pixels=1, aspect_log2_width=0.01),
        MODULE.Strategy("valid", "fixed", resolution_pixels=128, aspect_log2_width=1.0),
    )
    monkeypatch.setattr(MODULE, "STRATEGIES", strategies)

    output = tmp_path / "mixed"
    MODULE.run_geometry_analysis(root, output, scope="original", image_size=32, cv_folds=3)

    summary = pd.read_csv(output / "geometry_bin_summary.csv").set_index("strategy")
    assert summary.loc["empty", "selected_total"] == 0
    assert summary.loc["valid", "selected_total"] == 12
    assert summary.loc["empty", "candidate_assessment"] == "not_validated"

    probes = pd.read_csv(output / "geometry_bin_probe_results.csv")
    empty_probe = probes[probes["strategy"] == "empty"]
    valid_probe = probes[probes["strategy"] == "valid"]
    assert set(empty_probe["probe_status"]) == {"not_run"}
    assert set(empty_probe["reason"]) == {"empty_cohort"}
    assert set(valid_probe["probe_status"]) == {"ran"}

    distributions = pd.read_csv(output / "geometry_bin_distributions.csv")
    empty_distribution = distributions[
        (distributions["strategy"] == "empty") & (distributions["population"] == "candidate")
    ]
    assert set(empty_distribution["status"]) == {"empty_cohort"}
    assert (empty_distribution["count"] == 0).all()

"""Read-only metadata diagnostic tests using a temporary synthetic collection."""

from __future__ import annotations

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_metadata_shortcut.py"
SPEC = spec_from_file_location("diagnose_metadata_shortcut", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _encoded(size: tuple[int, int], fmt: str, quality: int = 90) -> bytes:
    stream = BytesIO()
    image = Image.new("RGB", size, (40, 80, 120))
    if fmt == "JPEG":
        image.save(stream, format=fmt, quality=quality)
    else:
        image.save(stream, format=fmt)
    image.close()
    return stream.getvalue()


def _collection(tmp_path: Path) -> tuple[Path, str]:
    data_root = tmp_path / "data"
    main = data_root / "main"
    rows = []
    specs = (
        [(0, (64, 64), "JPEG") for _ in range(4)]
        + [(0, (80, 40), "JPEG") for _ in range(2)]
        + [(1, (64, 64), "PNG") for _ in range(2)]
        + [(1, (96, 48), "PNG") for _ in range(4)]
    )
    for index, (label, size, fmt) in enumerate(specs):
        identifier = f"row_{index:03d}"
        relative = Path("real" if label == 0 else "ai") / f"{identifier}.{fmt.lower()}"
        path = main / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _encoded(size, fmt)
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
    manifest = main / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    (main / "acquisition.json").write_text(
        json.dumps({"status": "complete", "manifest_sha256": "fixture"}),
        encoding="utf-8",
    )
    return data_root, sha256(manifest.read_bytes()).hexdigest()


def test_diagnostic_writes_outputs_without_touching_data(tmp_path):
    data_root, manifest_hash = _collection(tmp_path)
    output = tmp_path / "diagnostic"
    result = MODULE.run_diagnostic(
        data_root,
        output,
        image_size=32,
        seed=42,
        cv_folds=3,
    )

    assert result["make_splits_called"] is False
    assert sha256((data_root / "main/manifest.csv").read_bytes()).hexdigest() == manifest_hash
    assert not (data_root / "splits.csv").exists()
    expected = {
        "metadata_distributions.csv",
        "dimension_frequencies.csv",
        "metadata_ablation.csv",
        "metadata_feature_ranking.csv",
        "geometry_matching_by_key.csv",
        "geometry_matching_summary.csv",
        "diagnostic_run.json",
        "metadata_shortcut_diagnostic.md",
    }
    assert {path.name for path in output.iterdir()} == expected

    dimensions = pd.read_csv(output / "dimension_frequencies.csv")
    assert set(dimensions["width"]) == {64, 80, 96}
    matching = pd.read_csv(output / "geometry_matching_summary.csv")
    eligible = matching[matching["population"] == "eligible"]
    assert set(eligible["post_match_balance"]) == {True}
    assert eligible["matched_per_class"].max() == 2

    ablation = pd.read_csv(output / "metadata_ablation.csv")
    assert set(ablation["method"]) == {"full", "single_feature", "leave_one_out"}
    assert len(ablation) == 2 * (1 + 6 + 6)

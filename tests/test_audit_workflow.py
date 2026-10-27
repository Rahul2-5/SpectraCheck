"""End-to-end Phase 2 gates on tiny synthetic acquisition files, no network."""

import hashlib
from io import BytesIO
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from aidetect import audit_workflow as workflow
from aidetect.io_utils import load_config, sha256_of_file


@pytest.fixture
def acquisition(tmp_path):
    cfg = load_config("config.yaml")
    cfg["paths"] = {"data_root": str(tmp_path / "data"), "features_root": str(tmp_path / "features"),
                    "reports_root": str(tmp_path / "reports"), "artifacts_root": str(tmp_path / "artifacts")}
    cfg["image_size"] = 32
    cfg["dataset"]["minimum_ai_images"] = 20
    collection = tmp_path / "data/main"
    collection.mkdir(parents=True)
    rng, rows = np.random.default_rng(42), []
    for label in (0, 1):
        for index in range(60):
            image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
            identifier = f"{label}_{index:03}"
            relative = f"{label}/{identifier}.png"
            path = collection / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            rows.append({"id": identifier, "label": label, "rel_path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(), "n_bytes": len(payload),
                "generator": "SD15" if label else "Real", "source_split": "train", "config_hash": "phase1"})
    pd.DataFrame(rows).to_csv(collection / "manifest.csv", index=False)
    record = {"status": "complete", "manifest_sha256": sha256_of_file(collection / "manifest.csv"),
              "specification": {"kind": "main", "generator_name": "SD15"}}
    (collection / "acquisition.json").write_text(json.dumps(record))
    return cfg


def test_actual_audit_and_split_pipeline(acquisition):
    from pathlib import Path
    result = workflow.run_audit(acquisition)
    assert result["status"] == "passed" and not result["normalization_applied"]
    assert result["final_probe"]["max_accuracy"] == .5
    frozen = workflow.run_split_freeze(acquisition)
    assert frozen["crossing_groups"] == 0
    assert len(frozen["splits_sha256"]) == 64
    assert (Path(acquisition["paths"]["reports_root"]) / "audit_report.md").exists()
    assert len(list((Path(acquisition["paths"]["reports_root"]) / "figures").glob("*.png"))) == 3
    assert workflow.run_split_freeze(acquisition) == frozen
    with pytest.raises(ValueError, match="after sealing"):
        workflow.run_audit(acquisition)


@pytest.mark.parametrize("final_score,status", [(.5, "passed"), (.8, "blocked")])
def test_normalization_both_classes_and_reprobe_gate(acquisition, monkeypatch, final_score, status):
    from pathlib import Path
    # Supply JPEG real images so the preplanned median quality is defined.
    root = Path(acquisition["paths"]["data_root"])
    manifest_path = root / "main/manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"id": str})
    for index, row in manifest[manifest["label"] == 0].iterrows():
        path = root / "main" / row["rel_path"]
        with Image.open(path) as image:
            stream = BytesIO()
            image.save(stream, format="JPEG", quality=85)
        path.write_bytes(stream.getvalue())
        manifest.loc[index, "sha256"] = sha256_of_file(path)
    manifest.to_csv(manifest_path, index=False)
    record_path = root / "main/acquisition.json"
    record = json.loads(record_path.read_text())
    record["manifest_sha256"] = sha256_of_file(manifest_path)
    record_path.write_text(json.dumps(record))
    scores = iter([.99, final_score])
    monkeypatch.setattr(workflow, "shortcut_probe", lambda *a, **k: {"max_accuracy": next(scores)})
    result = workflow.run_audit(acquisition)
    assert result["status"] == status and result["target_jpeg_quality"] == 85
    active = pd.read_csv(root / "audit/active_manifest.csv")
    assert (active["format"] == "JPEG").all() and (active["jpeg_quality"] == 85).all()
    assert len(list((root / "audit/normalized/real").glob("*.jpg"))) == 60
    assert len(list((root / "audit/normalized/ai").glob("*.jpg"))) == 60
    assert (root / "main/1/1_000.png").read_bytes().startswith(b"\x89PNG")
    if status == "blocked":
        with pytest.raises(ValueError, match="passing audit"):
            workflow.run_split_freeze(acquisition)
        assert not (root / "splits.csv").exists()


def test_current_config_and_image_integrity_required_for_splits(acquisition):
    from pathlib import Path
    workflow.run_audit(acquisition)
    acquisition["seed"] += 1
    with pytest.raises(ValueError, match="current configuration"):
        workflow.run_split_freeze(acquisition)
    acquisition["seed"] -= 1
    path = Path(acquisition["paths"]["data_root"]) / "main/0/0_000.png"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="image changed"):
        workflow.run_split_freeze(acquisition)


def test_undersized_exclusion_happens_before_freeze(acquisition):
    from pathlib import Path
    root = Path(acquisition["paths"]["data_root"])
    manifest_path = root / "main/manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"id": str})
    for label in (0, 1):
        path = root / f"main/{label}/{label}_000.png"
        Image.new("RGB", (16, 16), "red").save(path)
        manifest.loc[manifest["id"] == f"{label}_000", "sha256"] = sha256_of_file(path)
    manifest.to_csv(manifest_path, index=False)
    record_path = root / "main/acquisition.json"
    record = json.loads(record_path.read_text())
    record["manifest_sha256"] = sha256_of_file(manifest_path)
    record_path.write_text(json.dumps(record))
    result = workflow.run_audit(acquisition)
    assert result["excluded_rows"] == 2
    exclusions = pd.read_csv(root / "audit/excluded.csv")
    assert set(exclusions["reason"]) == {"smaller_than_target"}
    assert result["balanced_rows"] == 118


def test_missing_real_jpeg_quality_reports_block_without_inventing_fix(acquisition, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(workflow, "shortcut_probe", lambda *a, **k: {"max_accuracy": .9})
    result = workflow.run_audit(acquisition)
    assert result["status"] == "blocked" and result["target_jpeg_quality"] is None
    assert "No real-image JPEG quality" in result["remediation_error"]
    assert (Path(acquisition["paths"]["reports_root"]) / "audit_report.md").exists()
    with pytest.raises(ValueError, match="passing audit"):
        workflow.run_split_freeze(acquisition)

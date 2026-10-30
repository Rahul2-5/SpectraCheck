"""Candidate treatments must never change the official experiment."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from aidetect.io_utils import config_hash, load_config, sha256_of_file

SPEC = spec_from_file_location("fixed_crop_diagnostic", Path(__file__).parents[1] / "scripts/diagnose_fixed_crop.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def encoded(size=(20, 16), fmt="PNG"):
    image = Image.new("RGB", size, (30, 80, 130))
    stream = BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


def fixture(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "config.yaml")
    root = tmp_path / "data"
    cfg["paths"].update(data_root=str(root), artifacts_root=str(tmp_path / "artifacts"), features_root=str(tmp_path / "features"))
    cfg.update(image_size=8, cv_folds=2)
    rows = []
    for label in (0, 1):
        for index in range(4):
            ident = f"r{label}_{index}"
            relative = f"{ident}.png"
            payload = encoded((20 + index, 16 + index))
            path = root / "main" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            rows.append(dict(id=ident, label=label, rel_path=relative, sha256=sha256(payload).hexdigest()))
    pd.DataFrame(rows).to_csv(root / "main/manifest.csv", index=False)
    (root / "audit").mkdir()
    pd.DataFrame(rows)[["id", "label"]].to_csv(root / "audit/active_manifest.csv", index=False)
    audit = {"status": "blocked", "config_hash": config_hash(cfg), "target_jpeg_quality": 96,
        "active_manifest_sha256": sha256_of_file(root / "audit/active_manifest.csv"),
        "specification": {"source_manifest_sha256": sha256_of_file(root / "main/manifest.csv")}}
    (root / "audit/audit_result.json").write_text(json.dumps(audit))
    return cfg, root


def test_crop_is_fixed_without_resize_and_deterministic():
    image = Image.new("RGB", (20, 16), (255, 0, 0))
    image.paste((30, 80, 130), (6, 4, 14, 12))
    stream = BytesIO()
    image.save(stream, format="PNG")
    measured = MODULE.crop_metadata(stream.getvalue(), 8, 96)
    expected = MODULE.crop_metadata(encoded((8, 8)), 8, 96)
    assert measured == expected
    assert measured["width"] == measured["height"] == 8
    assert measured["jpeg_quality"] == 96


@pytest.mark.parametrize("payload,size,quality", [(encoded((4, 4)), 8, 96), (b"bad", 8, 96), (encoded(), 0, 96), (encoded(), 8, 101)])
def test_candidate_rejects_invalid_images_or_settings(payload, size, quality):
    with pytest.raises((ValueError, OSError)):
        MODULE.crop_metadata(payload, size, quality)


def test_candidate_reports_separately_and_preserves_originals(tmp_path):
    cfg, root = fixture(tmp_path)
    before = {p: sha256_of_file(p) for p in root.rglob("*") if p.is_file()}
    result = MODULE.run_candidate(cfg, tmp_path, tmp_path / "candidate")
    assert result["research_audit_status"] == "blocked"
    assert result["status"] == "exploratory_diagnostic_only"
    assert result["rows"] == 8
    assert not result["derived_images_persisted"]
    assert not result["research_splits_created"]
    assert result["candidate_probe"]["max_accuracy"] == .5
    assert before == {p: sha256_of_file(p) for p in root.rglob("*") if p.is_file()}
    assert sorted(p.name for p in (tmp_path / "candidate").iterdir()) == ["candidate_metadata.csv", "candidate_result.json"]


@pytest.mark.parametrize("kind", ["sealed", "source", "manifest", "label", "output", "config"])
def test_candidate_refuses_invalid_provenance_and_protected_outputs(tmp_path, kind):
    cfg, root = fixture(tmp_path)
    output = tmp_path / "candidate"
    if kind == "sealed":
        (root / "splits.csv").write_text("sealed")
    elif kind == "source":
        (root / "main/r0_0.png").write_bytes(b"changed")
    elif kind == "manifest":
        with (root / "main/manifest.csv").open("a") as stream:
            stream.write("changed")
    elif kind == "label":
        frame = pd.read_csv(root / "audit/active_manifest.csv")
        frame.loc[0, "label"] = 1
        frame.to_csv(root / "audit/active_manifest.csv", index=False)
        path = root / "audit/audit_result.json"
        audit = json.loads(path.read_text())
        audit["active_manifest_sha256"] = sha256_of_file(root / "audit/active_manifest.csv")
        path.write_text(json.dumps(audit))
    elif kind == "output":
        output = root / "candidate"
    elif kind == "config":
        cfg["seed"] = 43
    with pytest.raises(ValueError):
        MODULE.run_candidate(cfg, tmp_path, output)

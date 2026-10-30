"""R2 known-answer treatment, provenance and split/extraction integration."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from aidetect.audit import audit_table
from aidetect.audit_workflow import run_audit, run_split_freeze
from aidetect.feature_extraction import _source_payload
from aidetect.features import power_spectrum, rapsd
from aidetect.io_utils import load_config, sha256_of_file
from aidetect.preprocess import _canonical_patch, preprocess_image
import aidetect.r2_audit as module
from app.backend.inference import InferenceService


@pytest.fixture
def setup_r2(tmp_path, monkeypatch):
    cfg = load_config("config_r2.yaml")
    source, root = tmp_path / "legacy", tmp_path / "r2"
    cfg["paths"].update(data_root=str(root), artifacts_root=str(tmp_path / "artifacts"),
        features_root=str(tmp_path / "features"), reports_root=str(tmp_path / "reports"))
    cfg.update(image_size=32, cv_folds=2)
    cfg["dataset"]["minimum_ai_images"] = 4
    cfg["splitting"]["proportion_tolerance"] = .1
    cfg["r2"]["source_data_root"] = str(source)
    main = source / "main"
    main.mkdir(parents=True)
    rows = []
    for label in (0, 1):
        for i in range(8):
            ident = f"i{label}_{i}"
            a = np.random.default_rng(label * 100 + i).integers(0, 256, (40 + i, 45 + i, 3), dtype=np.uint8)
            p = main / f"{ident}.png"
            Image.fromarray(a).save(p)
            rows.append(dict(id=ident, label=label, rel_path=p.name,
                sha256=sha256_of_file(p), config_hash="legacy"))
    frame = pd.DataFrame(rows)
    frame.to_csv(main / "manifest.csv", index=False)
    out = source / "audit"
    out.mkdir()
    measured = audit_table(frame, main)
    measured.to_csv(out / "original_audit.csv", index=False)
    frame.to_csv(out / "active_manifest.csv", index=False)
    raw = {"status": "blocked", "target_jpeg_quality": 96,
        "specification": {"source_manifest_sha256": sha256_of_file(main / "manifest.csv")},
        "active_manifest_sha256": sha256_of_file(out / "active_manifest.csv")}
    (out / "audit_result.json").write_text(json.dumps(raw))
    cfg["r2"]["source_audit_sha256"] = sha256_of_file(out / "audit_result.json")
    monkeypatch.setattr(module, "shortcut_probe", lambda *args, **kw: {"max_accuracy": .5})
    return cfg, source, root


def test_r2_crop_matches_known_patch_and_shared_inference(tmp_path):
    cfg = load_config("config_r2.yaml")
    cfg.update(image_size=32, rapsd_rings=8)
    a = np.random.default_rng(42).integers(0, 256, (41, 51, 3), dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(a).save(stream, format="PNG")
    raw = stream.getvalue()
    expected = BytesIO()
    Image.fromarray(a[4:36, 9:41]).save(expected, format="JPEG", quality=96, subsampling=0, optimize=False, progressive=False)
    assert _canonical_patch(raw, cfg) == expected.getvalue()
    with Image.open(BytesIO(expected.getvalue())) as decoded:
        np.testing.assert_array_equal(preprocess_image(raw, cfg), np.asarray(decoded.convert("L"), dtype=np.float64) / 255.)
    (tmp_path / "config_r2.yaml").write_text(yaml.safe_dump(cfg))
    service = InferenceService(tmp_path, {"research_config": "config_r2.yaml"})
    assert service.config_path == tmp_path / "config_r2.yaml"
    analysis = service._feature_analysis(raw, cfg)
    expected_power = power_spectrum(preprocess_image(raw, cfg), window=cfg["frequency"]["hann_window"])
    np.testing.assert_array_equal(analysis["rapsd"]["log_power"], rapsd(expected_power, 8, epsilon=cfg["frequency"]["epsilon"]))
    assert "fixed centre patch" in analysis["disclaimer"]
    with pytest.raises(ValueError):
        InferenceService(tmp_path, {"research_config": "../external.yaml"})
    with pytest.raises(ValueError, match="smaller"):
        preprocess_image(expected.getvalue(), {**cfg, "image_size": 64})


def test_r2_audit_preserves_raw_sources_and_binds_originals(setup_r2, tmp_path):
    cfg, source, root = setup_r2
    hashes = {p: sha256_of_file(p) for p in source.rglob("*") if p.is_file()}
    result = run_audit(cfg)
    assert result["status"] == "passed" and result["raw_audit_status"] == "blocked"
    active = pd.read_csv(root / "audit/active_manifest.csv")
    assert active.metadata_domain.eq("treated_input").all()
    row = active.iloc[0].to_dict()
    payload = _source_payload(row, root, cfg)
    assert sha256(payload).hexdigest() == row["sha256"]
    assert sha256(_canonical_patch(payload, cfg)).hexdigest() == row["canonical_sha256"]
    assert not list((root / "audit").glob("*.jpg"))
    assert hashes == {p: sha256_of_file(p) for p in source.rglob("*") if p.is_file()}
    split = run_split_freeze(cfg)
    assert split["crossing_groups"] == 0
    with pytest.raises(ValueError):
        run_audit(cfg)
    with pytest.raises(ValueError, match="treated"):
        _source_payload({**row, "canonical_sha256": "changed"}, root, cfg)


@pytest.mark.parametrize("kind", ["overlap", "raw_hash", "image", "threshold", "sealed", "cohort"])
def test_r2_refuses_invalid_protocol_or_provenance(setup_r2, kind):
    cfg, source, root = setup_r2
    if kind == "overlap":
        cfg["paths"]["data_root"] = str(source)
    elif kind == "raw_hash":
        cfg["r2"]["source_audit_sha256"] = "changed"
    elif kind == "image":
        (source / "main/i0_0.png").write_bytes(b"changed")
    elif kind == "threshold":
        cfg["audit"]["shortcut_max_accuracy"] = .9
    elif kind == "sealed":
        root.mkdir()
        (root / "splits.csv").write_text("sealed")
    else:
        with (source / "audit/active_manifest.csv").open("a") as stream:
            stream.write("changed")
    with pytest.raises(ValueError):
        run_audit(cfg)


def test_r2_failed_probe_stays_blocked(setup_r2, monkeypatch):
    cfg, source, root = setup_r2
    monkeypatch.setattr(module, "shortcut_probe", lambda *args, **kw: {"max_accuracy": .8})
    assert run_audit(cfg)["status"] == "blocked"
    with pytest.raises(ValueError, match="passing"):
        run_split_freeze(cfg)
    with pytest.raises(ValueError, match="preserved"):
        run_audit(cfg)


def test_r2_interrupted_copy_resumes_but_never_replaces_changed_bytes(setup_r2, monkeypatch):
    cfg, source, root = setup_r2
    canonical = module._canonical_patch
    monkeypatch.setattr(module, "_canonical_patch", lambda *args: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        run_audit(cfg)
    assert json.loads((root / "audit/audit_result.json").read_text())["status"] == "incomplete"
    monkeypatch.setattr(module, "_canonical_patch", canonical)
    assert run_audit(cfg)["status"] == "passed"

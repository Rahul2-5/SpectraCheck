"""One connected Phases 2-8 run on generated images in a temporary repository.

No downloads, real model, real test rows, or project evaluation artifacts.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from scipy.ndimage import gaussian_filter
import yaml

from aidetect.audit_workflow import run_audit, run_split_freeze
from aidetect.evaluate import evaluate_test_once
from aidetect.freeze import freeze_phase7
from aidetect.io_utils import config_hash, load_config, sha256_of_file
from aidetect.train import run_phase6
from app.backend.inference import InferenceService
from app.backend.main import create_app

PROJECT = Path(__file__).parents[1]


def test_generated_images_to_evaluation_and_api(tmp_path, monkeypatch):
    repo = tmp_path / "pipeline"
    repo.mkdir()
    shutil.copytree(PROJECT / "src/aidetect", repo / "src/aidetect", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJECT / "scripts", repo / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / ".gitignore").write_text("data/\nfeatures/\nartifacts/models/\n__pycache__/\n")
    cfg = load_config(PROJECT / "config.yaml")
    cfg.update(image_size=32, rapsd_rings=8, pixel_baseline_size=8, cv_folds=3, search_iters=1, batch_size=30)
    cfg["dataset"]["minimum_ai_images"] = 20
    cfg["extraction"]["peak_k_mad"] = [3]
    cfg["training"]["r0"] = {"min": 8, "max": 8, "step": 8}
    cfg["training"]["search_space"]["rf"].update(n_estimators=[8, 10], min_samples_leaf=[1, 2], max_depth=[None, 3])
    cfg["training"]["search_space"]["xgb"].update(n_estimators=[8, 10], max_depth=[2, 3])
    cfg["paths"] = {"data_root": str(repo / "data"), "features_root": str(repo / "features"),
                    "artifacts_root": "artifacts", "reports_root": "reports"}
    config_path = repo / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    collection = repo / "data/main"
    collection.mkdir(parents=True)
    rng, manifest = np.random.default_rng(43), []
    for label in (0, 1):
        directory = collection / str(label)
        directory.mkdir()
        for index in range(60):
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            if label == 0:
                pixels = gaussian_filter(pixels.astype(float), sigma=(1.7, 1.7, 0)).astype(np.uint8)
            path = directory / f"{label}_{index:03}.png"
            # Fixed uncompressed encoding prevents a metadata shortcut in this fixture.
            Image.fromarray(pixels).save(path, compress_level=0)
            manifest.append({"id": path.stem, "label": label, "rel_path": path.relative_to(collection).as_posix(),
                "generator": "SD15" if label else "Real", "source_split": "train", "n_bytes": path.stat().st_size,
                "sha256": sha256_of_file(path), "config_hash": config_hash(cfg)})
    manifest_path = collection / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    (collection / "acquisition.json").write_text(json.dumps({"status": "complete", "synthetic_fixture": True,
        "manifest_sha256": sha256_of_file(manifest_path), "specification": {"kind": "main", "generator_name": "SD15"}}))
    monkeypatch.chdir(repo)
    audit = run_audit(cfg)
    assert audit["status"] == "passed"
    frozen = run_split_freeze(cfg)
    for split in ("train", "val", "test"):
        completed = subprocess.run([sys.executable, str(repo / "scripts/extract_features.py"),
            "--config", str(config_path), "--split", split], capture_output=True, text=True, check=True)
        assert json.loads(completed.stdout)["rows"] > 0
    trained = run_phase6(cfg, config_file=config_path)
    assert len(trained["models"]) == 6 and trained["test_features_read"] is False
    assert not (repo / "artifacts/FREEZE.lock").exists()
    (repo / "reports/practical_target.md").write_text(
        "# Synthetic protocol target\n\nStatus: Agreed\n\nMinimum validation accuracy: 0.75\n"
        "Rationale: Generated fixture only, not a research result.\n"
        "Agreed by: Fixture member A, Fixture member B\nSupervisor: Fixture supervisor\nDecision date: 2026-10-28\n")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.STDOUT, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    git("add", ".")
    git("commit", "-m", "Synthetic fixture before freeze")
    freeze_phase7(cfg, config_file=config_path, repo_root=repo)
    git("add", "artifacts/FREEZE.lock")
    git("commit", "-m", "Synthetic fixture freeze")
    evaluated = evaluate_test_once(cfg, config_file=config_path, repo_root=repo)
    assert len(evaluated["models"]) == 6
    git("add", "artifacts/EVALUATION_STARTED.json", "reports/results.json", "reports/results.md", "reports/test_predictions.csv")
    git("commit", "-m", "Synthetic one-shot results")
    with pytest.raises(FileExistsError, match="single-use"):
        evaluate_test_once(cfg, config_file=config_path, repo_root=repo)
    primary = trained["primary_model"]["artifact_key"]
    predictions = pd.read_csv(repo / "reports/test_predictions.csv")
    # Match only the selected model's existing prediction, never reopen feature rows.
    selected = predictions[(predictions["feature_set"] == "frequency") &
                           (predictions["model"] == trained["primary_model"]["model"])].iloc[0]
    active = pd.read_csv(repo / "data/audit/active_manifest.csv", dtype={"id": str}).set_index("id")
    image_path = repo / "data" / active.loc[str(selected["id"]), "rel_path"]
    settings = load_config(PROJECT / "app/backend/settings.yaml")
    service = InferenceService(repo, settings)
    client = TestClient(create_app(repo, service=service, settings=settings))
    response = client.post("/api/analyze", content=image_path.read_bytes())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model"] == primary
    assert result["prediction"] == int(selected["pred"])
    assert result["ai_probability"] == pytest.approx(float(selected["prob"]), abs=1e-12)
    assert sha256_of_file(repo / "data/splits.csv") == frozen["splits_sha256"]

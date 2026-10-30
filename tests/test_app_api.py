"""Phase 8 upload, artifact gates, and parity; synthetic artifacts only."""

from io import BytesIO
import json
from pathlib import Path

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sklearn.dummy import DummyClassifier
import yaml

from aidetect.evaluate import _positive_probabilities
from aidetect.feature_extraction import _row_from_summary, summarize_preprocessed_image
from aidetect.features import extract_frequency_features, power_spectrum, rapsd, spectral_entropy
from aidetect.io_utils import config_hash, load_config, sha256_of_file
from aidetect.preprocess import preprocess_image
from aidetect.train import make_frequency_matrix
from app.backend.inference import InferenceService, runtime_versions
from app.backend.main import create_app


@pytest.fixture
def bundle(tmp_path):
    cfg = load_config("config.yaml")
    cfg.update(image_size=32, rapsd_rings=8)
    cfg["paths"]["artifacts_root"] = "artifacts"
    root = tmp_path / "project"
    artifacts = root / "artifacts"
    (artifacts / "models").mkdir(parents=True)
    (root / "src/aidetect").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for name in ("preprocess.py", "features.py", "evaluate.py"):
        (root / "src/aidetect" / name).write_bytes((Path("src/aidetect") / name).read_bytes())
    record = {"config_sha256": config_hash(cfg), "splits_sha256": "synthetic-split"}
    primary = {**record, "feature_set": "frequency", "model": "rf", "artifact_key": "frequency_rf",
               "selection_metric": "validation_accuracy"}
    choice = {**record, "r0": 8, "k_mad": 3}
    for name, value in (("primary_model", primary), ("feature_choice", choice), ("versions", runtime_versions())):
        (artifacts / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    model = DummyClassifier(strategy="prior").fit(np.zeros((4, 12)), [0, 1, 1, 1])
    joblib.dump(model, artifacts / "models/frequency_rf.joblib")
    files = [root / "config.yaml", *artifacts.glob("*.json"), *artifacts.glob("models/*"), *root.glob("src/aidetect/*")]
    lock = {"schema_version": 1, "config_sha256": config_hash(cfg), "primary_model": "frequency_rf",
            "splits_sha256": "synthetic-split", "hashes": {p.relative_to(root).as_posix(): sha256_of_file(p) for p in files}}
    (artifacts / "FREEZE.lock").write_text(json.dumps(lock), encoding="utf-8")
    settings = load_config("app/backend/settings.yaml")
    settings["max_upload_bytes"] = 50000
    stream = BytesIO()
    Image.fromarray(np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(stream, format="PNG")
    service = InferenceService(root, settings)
    return root, cfg, settings, service, stream.getvalue()


def test_api_probability_and_features_match_evaluation_path(bundle):
    root, cfg, settings, service, payload = bundle
    before = {p.relative_to(root): sha256_of_file(p) for p in root.rglob("*") if p.is_file()}
    client = TestClient(create_app(root, service=service, settings=settings))
    response = client.post("/api/analyze", content=payload, headers={"content-type": "image/png"})
    assert response.status_code == 200
    result = response.json()
    x = preprocess_image(payload, cfg)
    summary = summarize_preprocessed_image(x, cfg)
    row = _row_from_summary({"id": "synthetic", "label": 1, "group_id": "g", "split": "test", "rel_path": "unused"}, summary, cfg["extraction"]["peak_k_mad"])
    matrix, _ = make_frequency_matrix(pd.DataFrame([row]), cfg, r0=8, k_mad=3)
    np.testing.assert_allclose(matrix[0], extract_frequency_features(x, cfg, 8, 3), rtol=1e-12)
    model = joblib.load(root / "artifacts/models/frequency_rf.joblib")
    pred, prob = _positive_probabilities(model, matrix, "frequency_rf")
    assert result["prediction"] == int(pred[0])
    assert result["ai_probability"] == float(prob[0]) == .75
    assert result["label"] == "AI-generated" and len(result["rapsd"]["log_power"]) == 8
    assert result["spectrum_png"].startswith("data:image/png;base64,")
    assert before == {p.relative_to(root): sha256_of_file(p) for p in root.rglob("*") if p.is_file()}
    assert not (root / "artifacts/EVALUATION_STARTED.json").exists()


@pytest.mark.parametrize("payload,code", [(b"not a picture.jpg", 422), (b"", 422), (b"a" * 50001, 413)], ids=["corrupt", "empty", "too-large"])
def test_bad_uploads_return_clear_errors(bundle, payload, code):
    root, _, settings, service, _ = bundle
    response = TestClient(create_app(root, service=service, settings=settings)).post("/api/analyze", content=payload)
    assert response.status_code == code and isinstance(response.json()["detail"], str)


def test_small_and_unsupported_images(bundle):
    root, _, settings, service, _ = bundle
    client = TestClient(create_app(root, service=service, settings=settings))
    for format_name, size in (("PNG", 16), ("GIF", 64)):
        stream = BytesIO()
        Image.new("RGB", (size, size)).save(stream, format=format_name)
        response = client.post("/api/analyze", content=stream.getvalue())
        assert response.status_code == 422


def test_missing_model_and_changed_inputs_fail_closed(bundle):
    root, _, settings, service, payload = bundle
    client = TestClient(create_app(root, service=service, settings=settings))
    assert client.get("/api/health").json()["feature_analysis_available"] is True
    (root / "artifacts/models/frequency_rf.joblib").write_bytes(b"changed model")
    health = client.get("/api/health").json()
    assert health["ready"] is False and health["feature_analysis_available"] is True
    response = client.post("/api/analyze", content=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "feature_analysis"
    assert result["model_available"] is False
    assert result["prediction"] is None and result["ai_probability"] is None
    assert result["label"] is None and result["model"] is None
    assert result["feature_analysis"]["rapsd_ring_count"] == 8
    (root / "artifacts/FREEZE.lock").unlink()
    assert service.health()["ready"] is False


def test_feature_analysis_matches_shared_frequency_functions_without_model(bundle):
    root, cfg, settings, service, payload = bundle
    (root / "artifacts/FREEZE.lock").unlink()
    response = TestClient(create_app(root, service=service, settings=settings)).post(
        "/api/analyze", content=payload
    )
    assert response.status_code == 200
    result = response.json()
    image = preprocess_image(payload, cfg)
    power = power_spectrum(image, window=cfg["frequency"]["hann_window"])
    expected_entropy = spectral_entropy(power, epsilon=cfg["frequency"]["epsilon"])
    expected_rapsd = rapsd(power, cfg["rapsd_rings"], epsilon=cfg["frequency"]["epsilon"])
    assert result["feature_analysis"]["spectral_entropy"] == pytest.approx(expected_entropy)
    np.testing.assert_allclose(result["rapsd"]["log_power"], expected_rapsd)
    assert result["spectrum_png"].startswith("data:image/png;base64,")


def test_version_mismatch_warns_and_selected_model_is_required(bundle):
    root, _, _, service, _ = bundle
    version_file = root / "artifacts/versions.json"
    versions = runtime_versions()
    versions["python"] = "old-version"
    version_file.write_text(json.dumps(versions))
    lock_file = root / "artifacts/FREEZE.lock"
    lock = json.loads(lock_file.read_text())
    lock["hashes"]["artifacts/versions.json"] = sha256_of_file(version_file)
    lock_file.write_text(json.dumps(lock))
    health = service.health()
    assert health["ready"] and any("old-version" in text for text in health["warnings"])
    primary_file = root / "artifacts/primary_model.json"
    primary = json.loads(primary_file.read_text())
    primary["selection_metric"] = "test_accuracy"
    primary_file.write_text(json.dumps(primary))
    assert not service.health()["ready"]


def test_inference_runtime_error_is_a_clear_service_error(bundle, monkeypatch):
    root, _, settings, service, payload = bundle

    def failed_inference(_payload):
        raise RuntimeError("Internal model compatibility failure")

    monkeypatch.setattr(service, "analyze", failed_inference)
    client = TestClient(create_app(root, service=service, settings=settings))
    response = client.post("/api/analyze", content=payload)
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()
    assert "Internal model" not in response.json()["detail"]

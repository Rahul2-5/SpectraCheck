"""Phase 8 upload, artifact gates, and parity; synthetic artifacts only."""

from io import BytesIO
import json
from pathlib import Path
import shutil

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
from aidetect.features import extract_frequency_features, feature_names, power_spectrum, rapsd, spectral_entropy
from aidetect.io_utils import config_hash, load_config, sha256_of_file
from aidetect.preprocess import preprocess_image
from aidetect.train import make_frequency_matrix
from app.backend.demo import DEMO_NOTICE, DevelopmentDemoService
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
    settings["research_config"] = "config.yaml"
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


def test_verified_r2_frequency_svm_serving_contract(tmp_path):
    project = Path(__file__).parents[1]
    root = tmp_path / "r2-serving"
    for relative in (
        "config_r2_fast_rf_t4.yaml",
        "artifacts/r2_fast_rf_t4/feature_choice.json",
        "artifacts/r2_fast_rf_t4/primary_model.json",
        "artifacts/r2_fast_rf_t4/versions.json",
        "artifacts/r2_fast_rf_t4/models/frequency_svm.joblib",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, destination)
    for relative in ("preprocess.py", "features.py", "evaluate.py"):
        destination = root / "src/aidetect" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / "src/aidetect" / relative, destination)

    config = load_config(root / "config_r2_fast_rf_t4.yaml")
    primary = json.loads((root / "artifacts/r2_fast_rf_t4/primary_model.json").read_text())
    choice = json.loads((root / "artifacts/r2_fast_rf_t4/feature_choice.json").read_text())
    assert config_hash(config) == primary["config_sha256"]
    assert primary["artifact_key"] == "frequency_svm"
    assert (choice["r0"], choice["k_mad"]) == (120, 5)
    assert config["preprocess"]["version"] == 2
    assert config["preprocess"]["crop"] == "fixed_center_patch"
    assert len(feature_names(config)) == 68

    required = [
        root / "config_r2_fast_rf_t4.yaml",
        root / "artifacts/r2_fast_rf_t4/feature_choice.json",
        root / "artifacts/r2_fast_rf_t4/primary_model.json",
        root / "artifacts/r2_fast_rf_t4/versions.json",
        root / "artifacts/r2_fast_rf_t4/models/frequency_svm.joblib",
        *(root / "src/aidetect" / name for name in ("preprocess.py", "features.py", "evaluate.py")),
    ]
    lock = {
        "schema_version": 1,
        "config_sha256": config_hash(config),
        "primary_model": "frequency_svm",
        "splits_sha256": primary["splits_sha256"],
        "hashes": {path.relative_to(root).as_posix(): sha256_of_file(path) for path in required},
    }
    artifacts = root / "artifacts/r2_fast_rf_t4"
    (artifacts / "FREEZE.lock").write_text(json.dumps(lock), encoding="utf-8")

    model = joblib.load(artifacts / "models/frequency_svm.joblib")
    assert getattr(model, "n_features_in_", None) == 68
    settings = load_config(project / "app/backend/settings.yaml")
    service = InferenceService(root, settings)
    stream = BytesIO()
    Image.fromarray(np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(
        stream, format="PNG"
    )
    response = TestClient(create_app(root, service=service, settings=settings)).post(
        "/api/analyze", content=stream.getvalue()
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "frequency_svm"
    assert body["processed_size"] == 256
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["ai_probability"] <= 1.0
    assert len(body["rapsd"]["log_power"]) == 64


def test_explicit_demo_service_loads_verified_model_without_research_lock():
    project = Path(__file__).parents[1]
    settings = load_config(project / "app/backend/settings.yaml")
    manifest = json.loads((project / "reports/phase_6_fast_rf_t4_backup_hashes.json").read_text())
    settings.update(
        demo_mode=True,
        demo_model_path="artifacts/r2_fast_rf_t4/models/frequency_svm.joblib",
        demo_model_sha256=manifest["artifacts/models/frequency_svm.joblib"],
    )
    service = DevelopmentDemoService(project, settings)
    health = service.health()
    assert health["ready"] is True
    assert health["mode"] == "development_demo"
    assert health["research_authorized"] is False
    assert health["demo_notice"] == DEMO_NOTICE
    assert not (project / "artifacts/r2_fast_rf_t4/FREEZE.lock").exists()


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


def test_missing_or_changed_model_and_inputs_fail_closed(bundle):
    root, _, settings, service, payload = bundle
    client = TestClient(create_app(root, service=service, settings=settings))
    assert client.get("/api/health").json()["feature_analysis_available"] is True
    model_path = root / "artifacts/models/frequency_rf.joblib"
    model_path.unlink()
    health = client.get("/api/health").json()
    assert health["ready"] is False and health["feature_analysis_available"] is True
    model_path.write_bytes(b"changed model")
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

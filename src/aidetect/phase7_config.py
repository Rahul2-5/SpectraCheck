"""Phase 7 configuration and completed Phase 6 artifact resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import config_hash, load_config, resolve_project_path


DEFAULT_PHASE7_CONFIG_NAME = "config_r2_fast_rf_t4.yaml"
EXPECTED_PHASE6_CONFIG_SHA256 = "46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766f"
PHASE7_MODEL_KEYS = (
    "frequency_svm",
    "frequency_rf",
    "frequency_xgb",
    "pixel_svm",
    "pixel_rf",
    "pixel_xgb",
)


def phase7_config_path(
    repo_root: str | Path, config_file: str | Path | None = None
) -> Path:
    """Resolve the explicit Phase 7 config and fail clearly if absent."""
    root = Path(repo_root).resolve()
    requested = DEFAULT_PHASE7_CONFIG_NAME if config_file is None else config_file
    path = resolve_project_path(root, requested)
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 7 configuration is missing: {path}. "
            f"Pass --config {DEFAULT_PHASE7_CONFIG_NAME} or provide the reviewed config file."
        )
    return path


def load_phase7_config(
    repo_root: str | Path, config_file: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Load the selected Phase 7 config without silently falling back."""
    path = phase7_config_path(repo_root, config_file)
    cfg = load_config(path)
    if path.name == DEFAULT_PHASE7_CONFIG_NAME:
        actual_hash = config_hash(cfg)
        if actual_hash != EXPECTED_PHASE6_CONFIG_SHA256:
            raise ValueError(
                f"{path.name} has unexpected config hash {actual_hash}; "
                f"expected completed Phase 6 hash {EXPECTED_PHASE6_CONFIG_SHA256}."
            )
    return path, cfg


def phase7_paths(cfg: dict[str, Any], repo_root: str | Path) -> dict[str, Path]:
    """Resolve nested artifact/report roots from the supplied configuration."""
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration paths must be a mapping")
    return {
        "artifacts_root": resolve_project_path(repo_root, paths.get("artifacts_root", "artifacts")),
        "reports_root": resolve_project_path(repo_root, paths.get("reports_root", "reports")),
        "features_root": resolve_project_path(repo_root, paths.get("features_root", "features")),
        "data_root": resolve_project_path(repo_root, paths.get("data_root", "data")),
    }


def phase7_artifact_paths(cfg: dict[str, Any], repo_root: str | Path) -> dict[str, Path]:
    """Return all Phase 7 inputs/outputs under the configured nested roots."""
    paths = phase7_paths(cfg, repo_root)
    artifacts_root = paths["artifacts_root"]
    reports_root = paths["reports_root"]
    result = {
        **paths,
        "practical_target": reports_root / "practical_target.md",
        "freeze_lock": artifacts_root / "FREEZE.lock",
        "evaluation_started": artifacts_root / "EVALUATION_STARTED.json",
        "splits_digest": artifacts_root / "splits.sha256",
        "feature_choice": artifacts_root / "feature_choice.json",
        "primary_model": artifacts_root / "primary_model.json",
        "versions": artifacts_root / "versions.json",
        "training_results": artifacts_root / "training_results.json",
        "results_json": reports_root / "results.json",
        "results_markdown": reports_root / "results.md",
        "predictions": reports_root / "test_predictions.csv",
    }
    result.update(
        {
            f"model:{key}": artifacts_root / "models" / f"{key}.joblib"
            for key in PHASE7_MODEL_KEYS
        }
    )
    return result


def validate_phase7_artifacts(
    cfg: dict[str, Any], repo_root: str | Path
) -> dict[str, Path]:
    """Validate required target and model files before any Phase 7 operation."""
    paths = phase7_artifact_paths(cfg, repo_root)
    required = [
        paths["practical_target"],
        paths["splits_digest"],
        paths["feature_choice"],
        paths["primary_model"],
        paths["versions"],
        paths["training_results"],
    ]
    required.extend(paths[f"model:{key}"] for key in PHASE7_MODEL_KEYS)
    missing = [path for path in required if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Phase 7 artifact bundle is incomplete; restore the verified Phase 6 "
            f"artifacts before proceeding:\n{details}"
        )
    return paths


__all__ = [
    "DEFAULT_PHASE7_CONFIG_NAME",
    "EXPECTED_PHASE6_CONFIG_SHA256",
    "PHASE7_MODEL_KEYS",
    "load_phase7_config",
    "phase7_artifact_paths",
    "phase7_config_path",
    "phase7_paths",
    "validate_phase7_artifacts",
]

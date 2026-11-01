"""Explicit supervisor-demo inference without research authorization.

This service is intentionally separate from the production lock-gated path. It
loads only the designated Phase 6 frequency SVM, verifies its committed backup
hash and metadata contract, and labels every response as a development demo.
It never reads test batches or creates a freeze/evaluation artifact.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib

from aidetect.features import feature_names
from aidetect.io_utils import config_hash, load_config, sha256_of_file
from app.backend.inference import InferenceService, runtime_versions


DEMO_NOTICE = (
    "Development demonstration only. This prediction is not an authorized "
    "Phase 7 research result."
)
_DEMO_ERRORS = (OSError, ValueError, KeyError, TypeError, ImportError,
                pickle.UnpicklingError, EOFError, AttributeError)


class DevelopmentDemoService(InferenceService):
    """Serve the explicitly designated Phase 6 model without a freeze lock."""

    def __init__(self, root: str | Path, settings: dict):
        if settings.get("demo_mode") is not True:
            raise ValueError("DevelopmentDemoService requires explicit demo_mode=true")
        model_value = settings.get("demo_model_path")
        expected_hash = settings.get("demo_model_sha256")
        if not isinstance(model_value, str) or not isinstance(expected_hash, str):
            raise ValueError("The demo model path and SHA-256 must be explicitly configured")
        super().__init__(root, settings)
        self.demo_model_path = (self.root / model_value).resolve()
        self.demo_model_sha256 = expected_hash.lower()
        if not self.demo_model_path.is_relative_to(self.root):
            raise ValueError("The demo model must be inside the project")

    def _bundle(self) -> tuple:
        cfg = load_config(self.config_path)
        artifacts = (self.root / cfg["paths"]["artifacts_root"]).resolve()
        expected_model = (artifacts / "models/frequency_svm.joblib").resolve()
        if self.demo_model_path != expected_model:
            raise ValueError("The demo model path must be the designated frequency_svm artifact")
        if sha256_of_file(self.demo_model_path).lower() != self.demo_model_sha256:
            raise ValueError("The designated demo model hash does not match the verified manifest")

        records = {
            name: json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("feature_choice", "primary_model", "versions")
        }
        primary, choice = records["primary_model"], records["feature_choice"]
        digest = config_hash(cfg)
        if (primary.get("artifact_key") != "frequency_svm"
            or primary.get("feature_set") != "frequency"
            or primary.get("selection_metric") != "validation_accuracy"):
            raise ValueError("The demo requires the validation-selected frequency_svm")
        if any(record.get("config_sha256") != digest for record in (primary, choice)):
            raise ValueError("Demo metadata does not match the selected configuration")
        if choice.get("splits_sha256") != primary.get("splits_sha256"):
            raise ValueError("Demo metadata disagrees on the frozen split")

        identity = (
            digest,
            sha256_of_file(self.demo_model_path),
            sha256_of_file(artifacts / "feature_choice.json"),
            sha256_of_file(artifacts / "primary_model.json"),
            sha256_of_file(artifacts / "versions.json"),
        )
        with self._guard:
            if self._cache is None or self._cache[0] != identity:
                model = joblib.load(self.demo_model_path)
                if getattr(model, "n_features_in_", None) != len(feature_names(cfg)):
                    raise ValueError("The demo model feature count does not match the R2 feature contract")
                version_warnings = [
                    f"{name}: trained with {expected}, installed {runtime_versions().get(name, 'unknown')}"
                    for name, expected in records["versions"].items()
                    if runtime_versions().get(name) != expected
                ]
                self._cache = (identity, cfg, choice, primary, model, version_warnings)
            return self._cache

    def health(self) -> dict:
        result = super().health()
        result.update({
            "mode": "development_demo",
            "demo_mode": True,
            "research_authorized": False,
            "demo_notice": DEMO_NOTICE,
        })
        if result.get("ready"):
            result["message"] = DEMO_NOTICE
        return result

    def analyze(self, payload: bytes) -> dict:
        result = super().analyze(payload)
        if result.get("mode") == "feature_analysis":
            return result
        result.update({
            "mode": "development_demo",
            "demo_mode": True,
            "research_authorized": False,
            "demo_notice": DEMO_NOTICE,
        })
        return result


__all__ = ["DEMO_NOTICE", "DevelopmentDemoService"]

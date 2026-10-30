"""In-memory inference using the validation-selected, hash-verified model.

This serving adapter never reads research test rows, predictions, or metrics.
It checks serving inputs against the existing lock without invoking evaluation.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import pickle
from pathlib import Path
import platform
import threading
from typing import Any
import warnings

import joblib
import numpy as np
from PIL import Image
import sklearn
import xgboost

from aidetect.evaluate import _positive_probabilities
from aidetect.features import extract_frequency_features, power_spectrum, rapsd, spectral_entropy
from aidetect.io_utils import config_hash, load_config, sha256_of_file
from aidetect.preprocess import preprocess_image

DISCLAIMER = ("Model score for one Stable Diffusion v1.5 dataset; not proof of authenticity; "
              "may be wrong on other generators, screenshots or compressed images.")
_BUNDLE_ERRORS = (OSError, ValueError, KeyError, TypeError, ImportError,
                  pickle.UnpicklingError, EOFError, AttributeError)


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "numpy": np.__version__, "joblib": joblib.__version__}


def inspect_upload(payload: bytes, settings: dict) -> None:
    """Validate encoded bytes/header dimensions without a disk-backed upload."""
    if not payload:
        raise ValueError("Choose an image before starting analysis.")
    if len(payload) > settings["max_upload_bytes"]:
        raise ValueError("This image exceeds the upload size limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in settings["allowed_formats"]:
                    raise ValueError("Choose a JPEG, PNG, or WebP image.")
                if image.width * image.height > settings["max_image_pixels"]:
                    raise ValueError("This image has too many pixels. Choose a smaller image.")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("This image has too many pixels. Choose a smaller image.") from exc
    except OSError as exc:
        raise ValueError("This file could not be read as an image. Choose a valid JPEG, PNG, or WebP.") from exc


class InferenceService:
    """Load only local serving artifacts; fail closed on changed/missing hashes."""

    def __init__(self, root: str | Path, settings: dict):
        self.root = Path(root).resolve()
        self.settings = settings
        self._cache: tuple | None = None
        self._guard = threading.RLock()

    def _bundle(self) -> tuple:
        cfg_path = self.root / "config.yaml"
        cfg = load_config(cfg_path)
        artifacts = (self.root / cfg["paths"]["artifacts_root"]).resolve()
        if not artifacts.is_relative_to(self.root):
            raise ValueError("Serving artifacts must be inside the local project.")
        lock = json.loads((artifacts / "FREEZE.lock").read_text(encoding="utf-8"))
        if lock.get("schema_version") != 1 or lock.get("config_sha256") != config_hash(cfg):
            raise ValueError("Research configuration differs from the frozen model.")
        records = {name: json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))
                   for name in ("feature_choice", "primary_model", "versions")}
        primary, choice = records["primary_model"], records["feature_choice"]
        key = primary.get("artifact_key")
        if (key not in {"frequency_svm", "frequency_rf", "frequency_xgb"}
            or primary.get("feature_set") != "frequency"
            or primary.get("selection_metric") != "validation_accuracy"
            or lock.get("primary_model") != key):
            raise ValueError("A validation-selected frequency model is required.")
        if any(record.get("config_sha256") != config_hash(cfg) for record in (primary, choice)):
            raise ValueError("Model and feature selection provenance differ from the configuration.")
        if any(record.get("splits_sha256") != lock.get("splits_sha256") for record in (primary, choice)):
            raise ValueError("Serving selection records disagree on the frozen split.")
        model_path = artifacts / "models" / f"{key}.joblib"
        required = [cfg_path, model_path, *(artifacts / f"{name}.json" for name in records),
                    *(self.root / "src/aidetect" / name for name in ("preprocess.py", "features.py", "evaluate.py"))]
        hashes = lock.get("hashes", {})
        # Only package source and serving artifacts: no sealed-batch or split reads.
        required.extend(self.root / name for name in hashes if name.startswith("src/aidetect/") and name.endswith(".py"))
        for path in required:
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                raise ValueError("Invalid serving artifact path.")
            name = resolved.relative_to(self.root).as_posix()
            if hashes.get(name) != sha256_of_file(resolved):
                raise ValueError("A frozen serving input changed. Restore the reviewed model bundle.")
        identity = tuple(hashes[path.resolve().relative_to(self.root).as_posix()] for path in required)
        with self._guard:
            if self._cache is None or self._cache[0] != identity:
                version_warnings = [f"{name}: trained with {expected}, installed {runtime_versions().get(name, 'unknown')}"
                    for name, expected in records["versions"].items()
                    if runtime_versions().get(name) != expected]
                with warnings.catch_warnings(record=True) as caught:
                    model = joblib.load(model_path)
                version_warnings.extend(str(item.message) for item in caught)
                self._cache = (identity, cfg, choice, primary, model, version_warnings)
            return self._cache

    def _feature_analysis_config(self) -> dict[str, Any]:
        """Load and validate only the configuration needed for feature analysis."""
        cfg = load_config(self.root / "config.yaml")
        if not isinstance(cfg.get("image_size"), int) or cfg["image_size"] < 1:
            raise ValueError("Feature analysis requires a positive image_size.")
        frequency = cfg.get("frequency")
        if not isinstance(frequency, dict):
            raise ValueError("Feature analysis requires frequency settings.")
        rings = cfg.get("rapsd_rings")
        if not isinstance(rings, int) or rings < 1:
            raise ValueError("Feature analysis requires positive rapsd_rings.")
        return cfg

    def _feature_analysis(self, payload: bytes, cfg: dict[str, Any]) -> dict[str, Any]:
        """Return model-independent preprocessing and frequency diagnostics."""
        image = preprocess_image(payload, cfg)
        frequency = cfg["frequency"]
        power = power_spectrum(image, window=frequency["hann_window"])
        entropy = spectral_entropy(power, epsilon=frequency["epsilon"])
        rapsd_values = rapsd(power, cfg["rapsd_rings"], epsilon=frequency["epsilon"])

        log_power = np.log10(power + frequency["epsilon"])
        span = float(np.ptp(log_power))
        display = (
            np.zeros_like(log_power, dtype=np.uint8)
            if span == 0
            else np.rint((log_power - log_power.min()) / span * 255).astype(np.uint8)
        )
        buffer = BytesIO()
        Image.fromarray(display).save(buffer, format="PNG")
        rings = cfg["rapsd_rings"]
        edges = np.linspace(1.0, cfg["image_size"] // 2, rings + 1)
        return {
            "mode": "feature_analysis",
            "model_available": False,
            "label": None,
            "prediction": None,
            "ai_probability": None,
            "model": None,
            "processed_size": cfg["image_size"],
            "warnings": [],
            "disclaimer": DISCLAIMER,
            "feature_analysis": {
                "spectral_entropy": float(entropy),
                "rapsd_ring_count": int(len(rapsd_values)),
            },
            "spectrum_png": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "rapsd": {
                "radius": ((edges[:-1] + edges[1:]) / 2).tolist(),
                "log_power": rapsd_values.tolist(),
            },
        }

    def health(self) -> dict[str, Any]:
        cfg = load_config(self.root / "config.yaml")
        base = {"min_image_side": cfg["image_size"], "max_upload_bytes": self.settings["max_upload_bytes"],
                "disclaimer": DISCLAIMER}
        try:
            self._feature_analysis_config()
            feature_available = True
            feature_message = None
        except _BUNDLE_ERRORS as exc:
            feature_available = False
            feature_message = f"Feature analysis is unavailable: {exc}"
        try:
            _, _, _, primary, _, notes = self._bundle()
            return {**base, "ready": True, "model": primary["artifact_key"], "warnings": notes,
                    "feature_analysis_available": feature_available,
                    "feature_analysis_message": feature_message}
        except _BUNDLE_ERRORS:
            return {**base, "ready": False, "model": None, "warnings": [],
                    "feature_analysis_available": feature_available,
                    "feature_analysis_message": feature_message,
                    "message": "A reviewed model bundle is not available. See the local setup instructions."}

    def analyze(self, payload: bytes) -> dict[str, Any]:
        inspect_upload(payload, self.settings)
        try:
            _, cfg, choice, primary, model, notes = self._bundle()
        except _BUNDLE_ERRORS:
            return self._feature_analysis(payload, self._feature_analysis_config())
        image = preprocess_image(payload, cfg)
        values = extract_frequency_features(image, cfg, r0=choice["r0"], k_mad=choice["k_mad"])
        predicted, positive = _positive_probabilities(model, values.reshape(1, -1), primary["artifact_key"])
        power = power_spectrum(image, window=cfg["frequency"]["hann_window"])
        log_power = np.log10(power + cfg["frequency"]["epsilon"])
        span = float(np.ptp(log_power))
        display = np.zeros_like(log_power, dtype=np.uint8) if span == 0 else np.rint((log_power - log_power.min()) / span * 255).astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(display).save(buffer, format="PNG")
        rings = cfg["rapsd_rings"]
        edges = np.linspace(1.0, cfg["image_size"] // 2, rings + 1)
        return {"label": "AI-generated" if predicted[0] else "Real", "prediction": int(predicted[0]),
                "ai_probability": float(positive[0]), "model": primary["artifact_key"],
                "processed_size": cfg["image_size"], "warnings": notes, "disclaimer": DISCLAIMER,
                "spectrum_png": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                "rapsd": {"radius": ((edges[:-1] + edges[1:]) / 2).tolist(), "log_power": values[2:2 + rings].tolist()}}

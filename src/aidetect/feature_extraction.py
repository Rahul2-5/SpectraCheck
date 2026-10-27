"""Resumable Phase 5 feature summaries and the sealed-test reader."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from numbers import Integral
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aidetect.features import (
    _settings,
    peaks,
    power_spectrum,
    rapsd,
    spectral_entropy,
)
from aidetect.pixel_baseline import extract_pixel_features
from aidetect.preprocess import preprocess_image


SCHEMA_VERSION = 1
DEFAULT_PEAK_K_MAD = (3, 4, 5, 6, 8)
SUMMARY_ARRAY_PREFIXES = ("cumulative_energy_", "rapsd_ring_", "pixel_")


def _peak_values(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    extraction = cfg.get("extraction", {})
    if extraction is None:
        extraction = {}
    if not isinstance(extraction, Mapping):
        raise ValueError("cfg['extraction'] must be a mapping")
    values = extraction.get("peak_k_mad", DEFAULT_PEAK_K_MAD)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("cfg['extraction']['peak_k_mad'] must be a sequence")
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 or float(value) != value for value in result):
        raise ValueError("peak_k_mad values must be positive integers")
    if len(set(result)) != len(result):
        raise ValueError("peak_k_mad values must be unique")
    return result


def _batch_size(cfg: Mapping[str, Any]) -> int:
    value = cfg.get("batch_size", 500)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("cfg['batch_size'] must be a positive integer")
    return int(value)


def _radial_cumulative_energy(power: np.ndarray) -> np.ndarray:
    """Return non-DC cumulative energy indexed by ceil(Euclidean radius).

    For an integer cutoff ``r0``, ``ceil(radius) <= r0`` is equivalent to
    ``radius <= r0``.  The resulting lookup therefore exactly reproduces the
    direct energy-ratio masks, including corner bins outside the RAPSD range.
    """
    side = power.shape[0]
    coordinates = np.arange(side, dtype=np.float64) - side // 2
    radii = np.hypot(coordinates[:, None], coordinates[None, :])
    indices = np.ceil(radii).astype(np.int64)
    energy = np.bincount(
        indices.reshape(-1), weights=power.reshape(-1), minlength=int(indices.max()) + 1
    ).astype(np.float64)
    energy[0] = 0.0  # Exclude DC from every low/high energy lookup.
    return np.cumsum(energy, dtype=np.float64)


def energy_ratio_from_cumulative(
    cumulative_energy: Sequence[float], r0: int, epsilon: float = 1.0e-12
) -> float:
    """Recompute the direct non-DC energy ratio from a saved lookup."""
    if isinstance(r0, bool) or not isinstance(r0, Integral) or r0 < 0:
        raise ValueError("r0 must be a non-negative integer")
    floor = float(epsilon)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("epsilon must be a finite positive number")
    cumulative = np.asarray(cumulative_energy, dtype=np.float64)
    if cumulative.ndim != 1 or cumulative.size == 0 or not np.isfinite(cumulative).all():
        raise ValueError("cumulative_energy must be a finite non-empty vector")
    if np.any(np.diff(cumulative) < -1.0e-10):
        raise ValueError("cumulative_energy must be non-decreasing")
    low_index = min(int(r0), cumulative.size - 1)
    low = float(cumulative[low_index])
    total = float(cumulative[-1])
    value = (total - low) / (low + floor)
    if not np.isfinite(value):
        raise RuntimeError("Lookup energy ratio is not finite")
    return float(value)


def summarize_preprocessed_image(
    x: np.ndarray,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute all persisted summaries from one already-preprocessed image."""
    rings, hann_window, epsilon, mad_scale = _settings(cfg)
    peak_k_mad = _peak_values(cfg)
    power = power_spectrum(x, window=hann_window)
    cumulative_energy = _radial_cumulative_energy(power)
    rapsd_values = rapsd(power, rings, epsilon=epsilon)
    entropy = spectral_entropy(power, epsilon=epsilon)
    peak_counts: dict[int, int] = {}
    peak_strengths: dict[int, float] = {}
    for threshold in peak_k_mad:
        count, strength = peaks(
            power, threshold, epsilon=epsilon, mad_scale=mad_scale
        )
        peak_counts[threshold] = count
        peak_strengths[threshold] = strength
    pixel_values = extract_pixel_features(
        x, size=int(cfg.get("pixel_baseline_size", 32))
    )
    result = {
        "cumulative_energy": cumulative_energy,
        "rapsd": rapsd_values,
        "entropy": float(entropy),
        "peak_counts": peak_counts,
        "peak_strengths": peak_strengths,
        "pixel": pixel_values,
        "side": int(x.shape[0]),
    }
    if not np.isfinite(result["cumulative_energy"]).all() or not np.isfinite(
        result["rapsd"]
    ).all() or not np.isfinite(result["pixel"]).all():
        raise RuntimeError("Feature summary contains NaN or Inf")
    if not np.isfinite(result["entropy"]):
        raise RuntimeError("Feature summary entropy is not finite")
    return result


def _id_digest(ids: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _column_names(summary: Mapping[str, Any], peak_k_mad: Sequence[int]) -> list[str]:
    return [
        *[f"cumulative_energy_{index:03d}" for index in range(len(summary["cumulative_energy"]))],
        *[f"rapsd_ring_{index:02d}" for index in range(len(summary["rapsd"]))],
        "entropy",
        *[f"peak_count_k{value:g}" for value in peak_k_mad],
        *[f"peak_strength_k{value:g}" for value in peak_k_mad],
        *[f"pixel_{index:04d}" for index in range(len(summary["pixel"]))],
    ]


def _row_from_summary(
    identity: Mapping[str, Any],
    summary: Mapping[str, Any],
    peak_k_mad: Sequence[int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": str(identity["id"]),
        "label": int(identity["label"]),
        "group_id": str(identity.get("group_id", "")),
        "split": str(identity["split"]),
        "rel_path": str(identity["rel_path"]),
        "image_side": int(summary["side"]),
        "entropy": float(summary["entropy"]),
    }
    row.update(
        {
            f"cumulative_energy_{index:03d}": float(value)
            for index, value in enumerate(summary["cumulative_energy"])
        }
    )
    row.update(
        {
            f"rapsd_ring_{index:02d}": float(value)
            for index, value in enumerate(summary["rapsd"])
        }
    )
    row.update(
        {
            f"peak_count_k{value:g}": int(summary["peak_counts"][value])
            for value in peak_k_mad
        }
    )
    row.update(
        {
            f"peak_strength_k{value:g}": float(summary["peak_strengths"][value])
            for value in peak_k_mad
        }
    )
    row.update(
        {f"pixel_{index:04d}": float(value) for index, value in enumerate(summary["pixel"])}
    )
    return row


def _metadata(
    *,
    cfg_hash: str,
    splits_hash: str,
    split: str,
    batch_index: int,
    batch_ids_hash: str,
    summary: Mapping[str, Any],
    peak_k_mad: Sequence[int],
) -> dict[bytes, bytes]:
    return {
        b"aidetect_schema_version": str(SCHEMA_VERSION).encode("ascii"),
        b"aidetect_config_hash": str(cfg_hash).encode("ascii"),
        b"aidetect_splits_sha256": str(splits_hash).encode("ascii"),
        b"aidetect_split": str(split).encode("ascii"),
        b"aidetect_batch_index": str(batch_index).encode("ascii"),
        b"aidetect_batch_ids_sha256": str(batch_ids_hash).encode("ascii"),
        b"aidetect_image_side": str(summary["side"]).encode("ascii"),
        b"aidetect_rapsd_rings": str(len(summary["rapsd"])).encode("ascii"),
        b"aidetect_pixel_baseline_size": str(
            int(np.sqrt(len(summary["pixel"])))
        ).encode("ascii"),
        b"aidetect_peak_k_mad": json.dumps(list(peak_k_mad)).encode("ascii"),
    }


def _read_metadata(path: Path) -> dict[bytes, bytes]:
    metadata = pq.read_metadata(path).metadata or {}
    return dict(metadata)


def _validate_existing_batch(
    path: Path,
    *,
    cfg_hash: str,
    splits_hash: str,
    split: str,
    batch_index: int,
    batch_ids_hash: str,
) -> None:
    metadata = _read_metadata(path)
    expected = {
        b"aidetect_schema_version": str(SCHEMA_VERSION).encode("ascii"),
        b"aidetect_config_hash": str(cfg_hash).encode("ascii"),
        b"aidetect_splits_sha256": str(splits_hash).encode("ascii"),
        b"aidetect_split": str(split).encode("ascii"),
        b"aidetect_batch_index": str(batch_index).encode("ascii"),
        b"aidetect_batch_ids_sha256": str(batch_ids_hash).encode("ascii"),
    }
    mismatches = [key.decode("ascii") for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(
            f"Existing feature batch {path} has mismatched metadata: {', '.join(mismatches)}"
        )


def _write_batch(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    cfg_hash: str,
    splits_hash: str,
    split: str,
    batch_index: int,
    batch_ids_hash: str,
    summary: Mapping[str, Any],
    peak_k_mad: Sequence[int],
) -> None:
    frame = pd.DataFrame(rows)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        _metadata(
            cfg_hash=cfg_hash,
            splits_hash=splits_hash,
            split=split,
            batch_index=batch_index,
            batch_ids_hash=batch_ids_hash,
            summary=summary,
            peak_k_mad=peak_k_mad,
        )
    )
    table = table.replace_schema_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def feature_output_dir(features_root: str | Path, split: str) -> Path:
    """Return the normal or sealed output directory for a valid split."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    root = Path(features_root)
    return root / "test_sealed" if split == "test" else root / split


def extract_split_features(
    rows: pd.DataFrame,
    *,
    data_root: str | Path,
    features_root: str | Path,
    cfg: Mapping[str, Any],
    split: str,
    config_digest: str,
    splits_digest: str,
) -> dict[str, Any]:
    """Extract one split into resumable Parquet batches.

    For ``split='test'`` this function writes to ``test_sealed`` and never
    reads any existing feature rows.  Existing batch metadata is checked only
    to prove that a resumable output belongs to the same configuration,
    split, and deterministic ID batch.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    required = {"id", "label", "group_id", "rel_path", "split"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Feature rows are missing columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError(f"No rows supplied for split {split}")
    if set(rows["split"].astype(str)) != {split}:
        raise ValueError("All supplied rows must belong to the requested split")
    if rows["id"].astype(str).duplicated().any():
        raise ValueError("Feature rows must have unique IDs")
    batch_size = _batch_size(cfg)
    peak_k_mad = _peak_values(cfg)
    ordered = rows.copy()
    ordered["id"] = ordered["id"].astype(str)
    ordered = ordered.sort_values("id").reset_index(drop=True)
    output_dir = feature_output_dir(features_root, split)
    data_root_path = Path(data_root)
    written = skipped = 0
    timings: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(ordered), batch_size)):
        batch = ordered.iloc[start : start + batch_size]
        batch_ids_hash = _id_digest(batch["id"].tolist())
        path = output_dir / f"batch_{batch_index:05d}.parquet"
        started = time.perf_counter()
        if path.exists():
            _validate_existing_batch(
                path,
                cfg_hash=config_digest,
                splits_hash=splits_digest,
                split=split,
                batch_index=batch_index,
                batch_ids_hash=batch_ids_hash,
            )
            skipped += 1
            timings.append(
                {"batch_index": batch_index, "rows": len(batch), "status": "skipped", "seconds": 0.0}
            )
            continue

        rows_for_file: list[dict[str, Any]] = []
        first_summary: dict[str, Any] | None = None
        for record in batch.to_dict(orient="records"):
            image_path = data_root_path / str(record["rel_path"])
            summary = summarize_preprocessed_image(preprocess_image(image_path, cfg), cfg)
            if first_summary is None:
                first_summary = summary
            elif (
                summary["side"] != first_summary["side"]
                or len(summary["cumulative_energy"]) != len(first_summary["cumulative_energy"])
                or len(summary["rapsd"]) != len(first_summary["rapsd"])
                or len(summary["pixel"]) != len(first_summary["pixel"])
            ):
                raise ValueError("All images in a feature batch must share the same summary shape")
            rows_for_file.append(_row_from_summary(record, summary, peak_k_mad))
        assert first_summary is not None
        _write_batch(
            rows_for_file,
            path,
            cfg_hash=config_digest,
            splits_hash=splits_digest,
            split=split,
            batch_index=batch_index,
            batch_ids_hash=batch_ids_hash,
            summary=first_summary,
            peak_k_mad=peak_k_mad,
        )
        written += 1
        timings.append(
            {
                "batch_index": batch_index,
                "rows": len(batch),
                "status": "written",
                "seconds": round(time.perf_counter() - started, 6),
            }
        )
    return {
        "split": split,
        "rows": int(len(ordered)),
        "batches": int((len(ordered) + batch_size - 1) // batch_size),
        "written_batches": written,
        "skipped_batches": skipped,
        "output_dir": str(output_dir),
        "timings": timings,
    }


def read_feature_batches(
    features_root: str | Path,
    split: str,
    *,
    artifacts_root: str | Path = "artifacts",
) -> pd.DataFrame:
    """Read extracted rows, enforcing the Phase 5 test-set seal."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    if split == "test":
        lock = Path(artifacts_root) / "FREEZE.lock"
        if not lock.is_file():
            raise PermissionError(
                "Reading test_sealed features requires artifacts/FREEZE.lock"
            )
    directory = feature_output_dir(features_root, split)
    paths = sorted(directory.glob("batch_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No feature batches found under {directory}")
    frames = [pd.read_parquet(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if result["id"].astype(str).duplicated().any():
        raise ValueError("Feature batches contain duplicate IDs")
    return result

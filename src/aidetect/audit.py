"""Metadata shortcut checks, reversible format remediation, and duplicate groups.

Audit operations are not training preprocessing. Original acquisition files are
never overwritten. No frequency feature is computed here.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import re
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from aidetect.data import detect_image_extension
from aidetect.io_utils import sha256_of_file

PROBE_COLUMNS = ["is_jpeg", "jpeg_quality", "width", "height", "n_bytes", "aspect_ratio"]


@lru_cache(maxsize=1)
def _quality_reference_tables() -> dict[int, dict[int, list[int]]]:
    """Use the installed JPEG encoder's standard tables, avoiding guessed tables."""
    references = {}
    for quality in range(1, 101):
        buffer = BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="JPEG", quality=quality, subsampling=0)
        with Image.open(BytesIO(buffer.getvalue())) as image:
            references[quality] = image.quantization
    return references


def estimate_jpeg_quality(tables: dict[int, list[int]]) -> tuple[int, float]:
    """Nearest standard quality by table MAE, not proof of encoding history.

    Custom tables may have no exact match. Return the residual alongside the
    estimate so the report can show uncertainty. Break ties at the lower quality.
    """
    if not tables or any(len(values) != 64 for values in tables.values()):
        raise ValueError("Missing or malformed JPEG quantization tables")
    errors = {}
    for quality, reference in _quality_reference_tables().items():
        common = sorted(set(tables) & set(reference))
        if not common:
            continue
        errors[quality] = float(np.mean([np.abs(np.asarray(tables[key], dtype=float)
            - np.asarray(reference[key], dtype=float)).mean() for key in common]))
    if not errors:
        raise ValueError("JPEG has no comparable standard quantization table")
    quality = min(errors, key=lambda candidate: (errors[candidate], candidate))
    return quality, errors[quality]


def describe_file(path: str | Path, *, phash_size: int = 8) -> dict[str, Any]:
    """Fully decode to reject corrupt files; record original encoded metadata."""
    path = Path(path)
    with path.open("rb") as stream:
        extension = detect_image_extension(stream.read(8))
    with Image.open(path) as image:
        image.load()
        quality, residual = (estimate_jpeg_quality(image.quantization)
                             if extension == "jpg" else (None, None))
        oriented = ImageOps.exif_transpose(image)
        return {"format": "JPEG" if extension == "jpg" else "PNG", "ext": extension,
            "width": image.width, "height": image.height, "mode": image.mode,
            "n_bytes": path.stat().st_size, "jpeg_quality": quality,
            "jpeg_quality_table_mae": residual, "is_jpeg": int(extension == "jpg"),
            "aspect_ratio": image.width / image.height,
            "sha256": sha256_of_file(path),
            "phash": str(imagehash.phash(oriented, hash_size=phash_size))}


def audit_table(manifest: pd.DataFrame, data_root: str | Path, *, phash_size: int = 8) -> pd.DataFrame:
    """Join descriptions to a manifest, checking IDs, path containment and hashes."""
    required = {"id", "rel_path", "label", "sha256"}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise ValueError("Manifest needs non-empty id, rel_path, label, sha256 columns")
    if manifest["id"].isna().any() or manifest["id"].duplicated().any():
        raise ValueError("Manifest IDs must be unique and non-missing")
    if not manifest["id"].map(lambda value: isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None).all():
        raise ValueError("Manifest IDs must be safe filename identifiers")
    if not manifest["label"].isin([0, 1]).all():
        raise ValueError("Expected binary labels real=0 and AI=1")
    root = Path(data_root).resolve()
    rows = []
    for row in manifest.to_dict("records"):
        path = (root / row["rel_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest image path escapes its data root")
        description = describe_file(path, phash_size=phash_size)
        if description["sha256"] != row["sha256"]:
            raise ValueError(f"Image changed after acquisition: {row['id']}")
        rows.append({**row, **description})
    return pd.DataFrame(rows)


def shortcut_probe(df: pd.DataFrame, seed: int, *, cv_folds: int = 5,
                   tree_depth: int = 3, logistic_max_iter: int = 2000) -> dict[str, Any]:
    """Both probes use identical stratified folds and metadata only.

    Reject an imbalanced sample: otherwise chance is not 50% and a 0.60 bar
    would be misleading. Missing PNG quality is represented by -1 as preplanned.
    """
    counts = df["label"].value_counts().to_dict()
    if set(counts) != {0, 1} or counts[0] != counts[1] or min(counts.values()) < cv_folds:
        raise ValueError("Shortcut probe requires balanced classes with at least cv_folds images each")
    x = df[PROBE_COLUMNS].copy()
    x["jpeg_quality"] = x["jpeg_quality"].fillna(-1)
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Metadata probe has missing or non-finite input")
    folds = list(StratifiedKFold(cv_folds, shuffle=True, random_state=seed).split(x, df["label"]))
    models = {"logistic_regression": make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=logistic_max_iter, random_state=seed)),
        "decision_tree": DecisionTreeClassifier(max_depth=tree_depth, random_state=seed)}
    scores = {}
    for name, model in models.items():
        values = cross_val_score(model, x, df["label"], cv=folds, scoring="accuracy", error_score="raise")
        scores[name] = {"mean_accuracy": float(values.mean()), "fold_accuracies": values.tolist()}
    return {"models": scores, "max_accuracy": max(s["mean_accuracy"] for s in scores.values()),
            "cv_folds": cv_folds, "columns": PROBE_COLUMNS, "counts": {str(k): v for k, v in counts.items()}}


def normalise_format(img_bytes: bytes, target_quality: int) -> bytes:
    """Identical JPEG remediation for both classes, preserving the source bytes.

    Apply EXIF orientation before stripping metadata; encode RGB at 4:4:4.
    This is an audit-time encoding policy, not the later classifier preprocessing.
    """
    if not isinstance(target_quality, int) or not 1 <= target_quality <= 100:
        raise ValueError("JPEG target quality must be an integer from 1 to 100")
    with Image.open(BytesIO(img_bytes)) as image:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=target_quality, subsampling=0,
                 optimize=False, progressive=False)
    return buffer.getvalue()


def duplicate_groups(df: pd.DataFrame, max_distance: int = 4) -> pd.Series:
    """Union-find transitive groups from exact bytes and perceptual Hamming distance.

    If normalized, union relationships from original and normalized hashes.
    IDs are stable under row reordering; a group is named after its lowest ID.
    Pairwise Hamming checks use O(n^2) time and O(n) memory for this small subset.
    """
    if max_distance < 0 or df["id"].duplicated().any() or df.empty:
        raise ValueError("Duplicate grouping needs unique IDs and a non-negative distance")
    ids = df["id"].astype(str).tolist()
    parents = list(range(len(ids)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parents[max(first, second)] = min(first, second)

    for column in ("sha256", "original_sha256", "canonical_sha256"):
        if column in df:
            seen = {}
            for index, value in enumerate(df[column]):
                if value in seen:
                    union(index, seen[value])
                else:
                    seen[value] = index
    for column in ("phash", "original_phash"):
        if column not in df:
            continue
        values = [int(value, 16) for value in df[column]]
        for first, value in enumerate(values):
            for second in range(first):
                if (value ^ values[second]).bit_count() <= max_distance:
                    union(first, second)
    minimum = {}
    for index, identifier in enumerate(ids):
        root = find(index)
        minimum[root] = min(identifier, minimum.get(root, identifier))
    return pd.Series([f"g_{minimum[find(i)]}" for i in range(len(ids))], index=df.index, name="group_id")

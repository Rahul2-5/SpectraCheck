"""The single image preprocessing entry point used by the detector.

Every image must use :func:`preprocess_image` before frequency features,
pixel-baseline features, training, evaluation, or app inference.  The fixed
operations are recorded in ``PREPROCESS_VERSION`` and mirrored in
``config.yaml`` so the preprocessing contract is part of future artifact
configuration hashes.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ImageSource: TypeAlias = str | Path | bytes | bytearray | memoryview

PREPROCESS_VERSION = {
    "version": 1,
    "orientation": "exif_transpose",
    "color_conversion": "RGB_then_L",
    "crop": "center_square",
    "resize_filter": "LANCZOS",
    "undersized_policy": "reject",
    "output_dtype": "float64",
    "normalization_divisor": 255.0,
}

# Explicit opt-in; historical configurations retain version 1 behavior.
R2_PREPROCESS_VERSION = {
    "version": 2, "orientation": "exif_transpose", "color_conversion": "RGB_JPEG_then_L",
    "crop": "fixed_center_patch", "resize_filter": "none", "undersized_policy": "reject",
    "jpeg_quality": 96, "jpeg_subsampling": 0, "jpeg_optimize": False,
    "jpeg_progressive": False, "output_dtype": "float64", "normalization_divisor": 255.0,
}


def _target_size(cfg: Mapping[str, object]) -> int:
    """Validate and return the configured square output side length."""
    value = cfg.get("image_size")
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("cfg['image_size'] must be a positive integer")
    return int(value)


def _validate_preprocess_settings(cfg: Mapping[str, object]) -> None:
    """Reject a config that silently disagrees with the fixed pipeline."""
    configured = cfg.get("preprocess")
    if configured is not None and configured not in (PREPROCESS_VERSION, R2_PREPROCESS_VERSION):
        raise ValueError("cfg['preprocess'] does not match PREPROCESS_VERSION")


def _canonical_patch(src: ImageSource, cfg: Mapping[str, object]) -> bytes:
    """R2's sole RGB patch/JPEG treatment, shared with its audit measurement."""
    _validate_preprocess_settings(cfg)
    if cfg.get("preprocess") != R2_PREPROCESS_VERSION:
        raise ValueError("Canonical patch requires explicit R2 preprocessing")
    target = _target_size(cfg)
    try:
        with _open_source(src) as image:
            image.load()
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            width, height = rgb.size
            if min(width, height) < target:
                raise ValueError("Image smaller than target; R2 never upscales")
            left, top = (width - target) // 2, (height - target) // 2
            patch = rgb.crop((left, top, left + target, top + target))
            stream = BytesIO()
            settings = cfg["preprocess"]
            patch.save(stream, format="JPEG", quality=settings["jpeg_quality"],
                subsampling=settings["jpeg_subsampling"], optimize=settings["jpeg_optimize"],
                progressive=settings["jpeg_progressive"])
            return stream.getvalue()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Unable to decode image: corrupt or unsupported image") from exc


def _open_source(src: ImageSource) -> Image.Image:
    """Open a path or in-memory encoded bytes without guessing from suffixes."""
    if isinstance(src, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(src)))
    if isinstance(src, (str, Path)):
        return Image.open(src)
    raise TypeError("src must be an image path or encoded image bytes")


def _decode_to_grayscale(src: ImageSource) -> Image.Image:
    """Decode, fully load, orient, and convert an image to independent L data."""
    try:
        with _open_source(src) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            grayscale = oriented.convert("RGB").convert("L")
            grayscale.load()
            return grayscale
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "Unable to decode image: corrupt or unsupported image"
        ) from exc


def preprocess_image(src: ImageSource, cfg: Mapping[str, object]) -> np.ndarray:
    """Return one square float64 grayscale array in the inclusive range [0, 1].

    The operation order is fixed: Pillow decode, EXIF transpose, RGB then L,
    centered square crop, conditional downsize with LANCZOS, and normalization.
    Images whose centered crop is smaller than the configured target are
    rejected; they are never upscaled.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping containing image_size")
    _validate_preprocess_settings(cfg)
    target = _target_size(cfg)
    if cfg.get("preprocess") == R2_PREPROCESS_VERSION:
        # Always accept original source bytes. Never accept an audited patch cache
        # as input: that would silently introduce a second JPEG generation.
        with Image.open(BytesIO(_canonical_patch(src, cfg))) as image:
            return np.ascontiguousarray(np.asarray(image.convert("L"), dtype=np.float64) / 255.0)
    grayscale = _decode_to_grayscale(src)
    try:
        width, height = grayscale.size
        side = min(width, height)
        if side < target:
            raise ValueError(
                f"Image smaller than target: centered crop side {side} < {target}"
            )

        left = (width - side) // 2
        top = (height - side) // 2
        square = grayscale.crop((left, top, left + side, top + side))
        try:
            if side > target:
                square = square.resize(
                    (target, target), Image.Resampling.LANCZOS
                )
            array = np.asarray(square, dtype=np.float64)
        finally:
            square.close()
    finally:
        grayscale.close()

    array /= float(PREPROCESS_VERSION["normalization_divisor"])
    if (
        array.shape != (target, target)
        or array.dtype != np.float64
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise RuntimeError("Preprocessing produced an invalid finite [0, 1] array")
    return np.ascontiguousarray(array, dtype=np.float64)

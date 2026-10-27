"""The prescribed pixel-space comparison baseline."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from PIL import Image


PIXEL_BASELINE_VERSION = {
    "version": 1,
    "input": "preprocess_image_output",
    "resize_filter": "LANCZOS",
    "output_dtype": "float64",
    "flatten_order": "C",
}


def _validated_input(x: np.ndarray) -> np.ndarray:
    """Validate the shared preprocessor's finite normalized output contract."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 2:
        raise ValueError("x must be a square 2D array with side at least 2")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("x must contain finite values in the inclusive range [0, 1]")
    return np.ascontiguousarray(array, dtype=np.float64)


def _target_size(size: Any) -> int:
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
        raise ValueError("size must be a positive integer")
    return int(size)


def extract_pixel_features(x: np.ndarray, size: int = 32) -> np.ndarray:
    """Downsize a preprocessed image with LANCZOS and flatten it.

    ``x`` must be the output of :func:`aidetect.preprocess.preprocess_image`.
    This function deliberately does not decode, crop, orient, or normalize an
    image.  The only resize in the baseline is the Phase 5 prescribed
    comparison downsize from the shared preprocessed array.
    """
    array = _validated_input(x)
    target = _target_size(size)
    if target > array.shape[0]:
        raise ValueError("pixel baseline cannot upscale a preprocessed image")

    if array.shape[0] == target:
        result = array.reshape(-1).copy()
    else:
        image = Image.fromarray(array.astype(np.float32), mode="F")
        try:
            resized = image.resize((target, target), Image.Resampling.LANCZOS)
            try:
                result = np.asarray(resized, dtype=np.float64).reshape(-1).copy()
            finally:
                resized.close()
        finally:
            image.close()

    # Pillow's floating-point resampler is expected to remain in-range for a
    # normalized source, but clipping protects the feature contract from tiny
    # platform-specific ringing at the boundaries.
    result = np.clip(result, 0.0, 1.0)
    if result.shape != (target * target,) or not np.isfinite(result).all():
        raise RuntimeError("Pixel baseline produced an invalid feature vector")
    return np.ascontiguousarray(result, dtype=np.float64)

"""Handcrafted frequency-domain features for preprocessed square images."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


DEFAULT_EPSILON = 1.0e-12
DEFAULT_MAD_SCALE = 1.4826


def _square_array(values: np.ndarray, name: str, *, nonnegative: bool = False) -> np.ndarray:
    """Validate a finite square numeric array and return float64 data."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 2:
        raise ValueError(f"{name} must be a square 2D array with side at least 2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _epsilon(epsilon: Real) -> float:
    """Validate the positive numerical floor used by log and ratio features."""
    value = float(epsilon)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("epsilon must be a finite positive number")
    return value


def _ring_count(value: object) -> int:
    """Validate an integer number of radial rings."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("rapsd_rings must be a positive integer")
    return int(value)


def _settings(cfg: Mapping[str, object]) -> tuple[int, bool, float, float]:
    """Read frequency settings while keeping small synthetic configs usable."""
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping containing rapsd_rings")
    rings = _ring_count(cfg.get("rapsd_rings"))
    frequency = cfg.get("frequency", {})
    if frequency is None:
        frequency = {}
    if not isinstance(frequency, Mapping):
        raise ValueError("cfg['frequency'] must be a mapping")
    hann_window = frequency.get("hann_window", True)
    if not isinstance(hann_window, bool):
        raise ValueError("cfg['frequency']['hann_window'] must be boolean")
    epsilon = _epsilon(frequency.get("epsilon", DEFAULT_EPSILON))
    mad_scale = float(frequency.get("mad_scale", DEFAULT_MAD_SCALE))
    if not np.isfinite(mad_scale) or mad_scale <= 0.0:
        raise ValueError("cfg['frequency']['mad_scale'] must be positive and finite")
    return rings, hann_window, epsilon, mad_scale


def power_spectrum(
    x: np.ndarray,
    window: bool | np.ndarray | None = True,
) -> np.ndarray:
    """Return the centered squared FFT magnitude of a square image.

    The image mean is removed before an optional fixed 2D Hann window is
    applied. ``fftshift`` places the DC bin at ``(S // 2, S // 2)`` for the
    even-sided square arrays used by the project. A 2D array may be supplied
    as a custom window for deterministic tests; normal extraction uses the
    configured Hann boolean.
    """
    image = _square_array(x, "x")
    centered = image - image.mean()
    if isinstance(window, (bool, np.bool_)):
        if window:
            one_dimensional = np.hanning(image.shape[0])
            centered = centered * np.outer(one_dimensional, one_dimensional)
    elif window is not None:
        window_array = _square_array(window, "window")
        if window_array.shape != image.shape:
            raise ValueError("window must have the same square shape as x")
        centered = centered * window_array
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    if not np.isfinite(power).all():
        raise RuntimeError("FFT power spectrum is not finite")
    return np.ascontiguousarray(power, dtype=np.float64)


def radius_map(size: int) -> np.ndarray:
    """Return each FFT bin's Euclidean radius from the shifted DC centre."""
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 2:
        raise ValueError("size must be an integer of at least 2")
    side = int(size)
    center = side // 2
    coordinates = np.arange(side, dtype=np.float64) - center
    return np.hypot(coordinates[:, None], coordinates[None, :])


def _validated_power(power: np.ndarray) -> np.ndarray:
    return _square_array(power, "power", nonnegative=True)


def energy_ratio(power: np.ndarray, r0: Real, epsilon: Real = DEFAULT_EPSILON) -> float:
    """Return high-frequency energy divided by non-DC low-frequency energy."""
    spectrum = _validated_power(power)
    cutoff = float(r0)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("r0 must be a finite non-negative number")
    floor = _epsilon(epsilon)
    radii = radius_map(spectrum.shape[0])
    high = spectrum[radii > cutoff].sum(dtype=np.float64)
    low = spectrum[(radii > 0.0) & (radii <= cutoff)].sum(dtype=np.float64)
    value = float(high / (low + floor))
    if not np.isfinite(value):
        raise RuntimeError("Energy ratio is not finite")
    return value


def spectral_entropy(power: np.ndarray, epsilon: Real = DEFAULT_EPSILON) -> float:
    """Return normalized Shannon entropy of non-DC power bins in [0, 1]."""
    spectrum = _validated_power(power)
    floor = _epsilon(epsilon)
    center = spectrum.shape[0] // 2
    non_dc = np.ones(spectrum.shape, dtype=bool)
    non_dc[center, center] = False
    values = spectrum[non_dc]
    total = float(values.sum(dtype=np.float64))
    if total <= floor:
        return 0.0
    probabilities = values / total
    bins = probabilities.size
    if bins <= 1:
        return 0.0
    entropy = -np.sum(probabilities * np.log2(probabilities + floor), dtype=np.float64)
    normalized = float(entropy / np.log2(bins))
    return float(np.clip(normalized, 0.0, 1.0))


def _ring_edges(side: int, rings: int) -> np.ndarray:
    maximum = side // 2
    # The final edge is one past S//2, so the integer radius S//2 is included.
    return np.linspace(1.0, float(maximum + 1), rings + 1)


def _ring_masks(side: int, rings: int) -> list[np.ndarray]:
    radii = radius_map(side)
    edges = _ring_edges(side, rings)
    return [
        (radii >= edges[index]) & (radii < edges[index + 1])
        for index in range(rings)
    ]


def rapsd(power: np.ndarray, rings: int, epsilon: Real = DEFAULT_EPSILON) -> np.ndarray:
    """Return K log10 radial mean-power values from radius 1 through S//2."""
    spectrum = _validated_power(power)
    count = _ring_count(rings)
    floor = _epsilon(epsilon)
    values = []
    for mask in _ring_masks(spectrum.shape[0], count):
        ring = spectrum[mask]
        mean_power = float(ring.mean(dtype=np.float64)) if ring.size else 0.0
        values.append(np.log10(mean_power + floor))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise RuntimeError("RAPSD output is invalid")
    return result


def _local_maxima(values: np.ndarray) -> np.ndarray:
    """Mark bins no lower than every value in their 3x3 neighborhood."""
    padded = np.pad(values, 1, mode="constant", constant_values=-np.inf)
    maxima = np.ones(values.shape, dtype=bool)
    for row in range(3):
        for column in range(3):
            maxima &= values >= padded[row : row + values.shape[0], column : column + values.shape[1]]
    return maxima


def peaks(
    power: np.ndarray,
    k_mad: Real,
    epsilon: Real = DEFAULT_EPSILON,
    mad_scale: Real = DEFAULT_MAD_SCALE,
) -> tuple[int, float]:
    """Count robust local spectral peaks once per conjugate pair.

    Log power is normalized separately within each radius ring by its median
    and ``mad_scale * MAD``. The accepted half-plane is the upper half plus
    the positive horizontal axis, excluding DC, so real-image conjugate pairs
    are counted once.
    """
    spectrum = _validated_power(power)
    threshold = float(k_mad)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("k_mad must be a finite positive number")
    floor = _epsilon(epsilon)
    scale = float(mad_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("mad_scale must be a finite positive number")

    # Clamp numerical FFT residue to the zero-power floor before taking logs;
    # otherwise exact synthetic sinusoids can produce artificial MAD peaks.
    log_power = np.log(np.maximum(spectrum, floor))
    z_scores = np.full(log_power.shape, -np.inf, dtype=np.float64)
    radii = radius_map(spectrum.shape[0])
    maximum_radius = spectrum.shape[0] // 2
    for radius in range(1, maximum_radius + 1):
        mask = (radii >= radius) & (radii < radius + 1.0)
        values = log_power[mask]
        if values.size == 0:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        z_scores[mask] = (values - median) / (scale * mad + floor)

    center = spectrum.shape[0] // 2
    row_coordinates = np.arange(spectrum.shape[0]) - center
    column_coordinates = np.arange(spectrum.shape[1]) - center
    half_plane = (row_coordinates[:, None] < 0) | (
        (row_coordinates[:, None] == 0) & (column_coordinates[None, :] > 0)
    )
    eligible = (radii >= 1.0) & (radii <= maximum_radius) & half_plane
    candidate = eligible & _local_maxima(log_power) & (z_scores > threshold)
    count = int(candidate.sum())
    strength = float(z_scores[candidate].max()) if count else 0.0
    if not np.isfinite(strength):
        raise RuntimeError("Peak strength is not finite")
    return count, strength


def feature_names(cfg: Mapping[str, object]) -> list[str]:
    """Return stable names matching ``extract_frequency_features`` order."""
    rings, _, _, _ = _settings(cfg)
    return [
        "energy_ratio",
        "spectral_entropy",
        *[f"rapsd_ring_{index:02d}" for index in range(rings)],
        "peak_count",
        "peak_strength",
    ]


def extract_frequency_features(
    x: np.ndarray,
    cfg: Mapping[str, object],
    r0: Real,
    k_mad: Real,
) -> np.ndarray:
    """Extract the fixed-order frequency vector from one preprocessed image."""
    rings, hann_window, epsilon, mad_scale = _settings(cfg)
    power = power_spectrum(x, window=hann_window)
    count, strength = peaks(power, k_mad, epsilon=epsilon, mad_scale=mad_scale)
    values = np.asarray(
        [
            energy_ratio(power, r0, epsilon=epsilon),
            spectral_entropy(power, epsilon=epsilon),
            *rapsd(power, rings, epsilon=epsilon),
            count,
            strength,
        ],
        dtype=np.float64,
    )
    names = feature_names(cfg)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise RuntimeError("Frequency feature vector is invalid")
    return values

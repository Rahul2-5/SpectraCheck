"""Known-answer tests for the handcrafted frequency feature module."""

import numpy as np
import pytest

from aidetect.features import (
    energy_ratio,
    extract_frequency_features,
    feature_names,
    peaks,
    power_spectrum,
    radius_map,
    rapsd,
    spectral_entropy,
)


def _cfg(*, rings=64, hann_window=False):
    return {
        "rapsd_rings": rings,
        "frequency": {
            "hann_window": hann_window,
            "epsilon": 1.0e-12,
            "mad_scale": 1.4826,
        },
    }


def test_power_spectrum_centers_dc_and_known_cosine_radius():
    side = 256
    columns = np.arange(side, dtype=np.float64)[None, :]
    image = np.cos(2.0 * np.pi * 16.0 * columns / side)

    power = power_spectrum(np.repeat(image, side, axis=0), window=False)
    center = side // 2
    power[center, center] = 0.0
    maximum = np.unravel_index(np.argmax(power), power.shape)

    assert maximum in ((center, center - 16), (center, center + 16))
    assert radius_map(side)[maximum] == pytest.approx(16.0)


def test_hann_window_is_optional_and_constant_input_is_finite():
    image = np.ones((32, 32), dtype=np.float64)
    without_window = power_spectrum(image, window=False)
    with_window = power_spectrum(image, window=True)

    assert np.all(without_window == 0.0)
    assert np.all(with_window == 0.0)
    values = extract_frequency_features(image, _cfg(rings=16), r0=8, k_mad=3)
    assert values.shape == (20,)
    assert np.isfinite(values).all()


def test_energy_ratio_excludes_dc_and_uses_non_dc_denominator():
    power = np.zeros((8, 8), dtype=np.float64)
    center = 4
    power[center, center] = 100.0
    power[center, center + 1] = 2.0
    power[center, center + 3] = 8.0

    assert energy_ratio(power, 2, epsilon=1.0e-12) == pytest.approx(4.0)


def test_uniform_non_dc_power_has_entropy_near_one():
    power = np.ones((16, 16), dtype=np.float64)
    power[8, 8] = 0.0

    assert spectral_entropy(power) == pytest.approx(1.0, abs=1.0e-10)


def test_rapsd_has_exact_ring_count_ignores_corners_and_logs_power():
    power = np.zeros((16, 16), dtype=np.float64)
    radii = radius_map(16)
    power[(radii >= 1) & (radii < 5)] = 10.0

    result = rapsd(power, rings=4, epsilon=1.0e-12)

    assert result.shape == (4,)
    assert result[0] == pytest.approx(1.0)
    assert result[-1] == pytest.approx(-12.0)
    assert np.isfinite(result).all()


def test_peaks_counts_one_half_of_conjugate_cosine_pair():
    side = 64
    columns = np.arange(side, dtype=np.float64)[None, :]
    image = np.cos(2.0 * np.pi * 8.0 * columns / side)
    power = power_spectrum(np.repeat(image, side, axis=0), window=False)

    count, strength = peaks(power, k_mad=3)

    assert count == 1
    assert strength > 3.0


def test_white_noise_entropy_is_high_and_rapsd_is_not_steep():
    rng = np.random.default_rng(42)
    power = power_spectrum(rng.normal(size=(128, 128)), window=False)
    values = rapsd(power, rings=16)
    slope = np.polyfit(np.arange(values.size), values, 1)[0]

    assert spectral_entropy(power) > 0.90
    assert abs(slope) < 0.05


def test_smooth_gradient_has_low_high_frequency_energy_ratio():
    gradient = np.tile(np.linspace(0.0, 1.0, 128), (128, 1))
    power = power_spectrum(gradient, window=False)
    checkerboard = np.indices((128, 128)).sum(axis=0) % 2
    gradient_rapsd = rapsd(power, rings=16)
    gradient_slope = np.polyfit(np.arange(gradient_rapsd.size), gradient_rapsd, 1)[0]

    assert energy_ratio(power, r0=8) < 1.0
    assert gradient_slope < -0.05
    assert energy_ratio(power_spectrum(checkerboard, window=False), r0=8) > energy_ratio(power, r0=8)


def test_feature_vector_names_length_finiteness_and_determinism():
    rng = np.random.default_rng(123)
    image = rng.random((64, 64))
    cfg = _cfg(rings=64, hann_window=True)

    names = feature_names(cfg)
    first = extract_frequency_features(image, cfg, r0=8, k_mad=4)
    second = extract_frequency_features(image, cfg, r0=8, k_mad=4)

    assert len(names) == 68
    assert first.shape == (len(names),)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert names[:2] == ["energy_ratio", "spectral_entropy"]
    assert names[-2:] == ["peak_count", "peak_strength"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: radius_map(1),
        lambda: power_spectrum(np.zeros((8, 4))),
        lambda: energy_ratio(np.zeros((8, 8)), -1),
        lambda: spectral_entropy(np.full((8, 8), np.nan)),
        lambda: rapsd(np.zeros((8, 8)), rings=0),
        lambda: peaks(np.zeros((8, 8)), k_mad=0),
    ],
)
def test_invalid_feature_inputs_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()

import numpy as np
import pytest

from aidetect.pixel_baseline import extract_pixel_features


def test_pixel_baseline_has_1024_finite_values_and_preserves_constant_input():
    image = np.full((64, 64), 0.37, dtype=np.float64)

    values = extract_pixel_features(image)

    assert values.shape == (1024,)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, 0.37, atol=2e-6)


def test_pixel_baseline_uses_the_requested_smaller_size():
    image = np.arange(64 * 64, dtype=np.float64).reshape(64, 64) / (64 * 64 - 1)

    values = extract_pixel_features(image, size=8)

    assert values.shape == (64,)
    assert np.all((values >= 0.0) & (values <= 1.0))


@pytest.mark.parametrize(
    "image,size,match",
    [
        (np.zeros((32, 32)), 64, "cannot upscale"),
        (np.zeros((32, 32)), 0, "positive integer"),
        (np.full((32, 32), np.nan), 8, "finite"),
        (np.zeros((16, 32)), 8, "square"),
    ],
)
def test_pixel_baseline_rejects_invalid_contract(image, size, match):
    with pytest.raises(ValueError, match=match):
        extract_pixel_features(image, size=size)

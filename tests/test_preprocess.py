"""Synthetic known-answer and failure-path checks for shared preprocessing."""

from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from aidetect.io_utils import config_hash, load_config
from aidetect.preprocess import PREPROCESS_VERSION, preprocess_image


def _encoded_image(image: Image.Image, *, format_name: str = "PNG", **kwargs) -> bytes:
    stream = BytesIO()
    image.save(stream, format=format_name, **kwargs)
    return stream.getvalue()


def test_output_shape_dtype_range_and_determinism_for_path_and_bytes(tmp_path):
    source = Image.new("RGB", (40, 30), "#336699")
    payload = _encoded_image(source)
    path = tmp_path / "sample.png"
    path.write_bytes(payload)
    cfg = {"image_size": 16}

    from_path = preprocess_image(path, cfg)
    from_bytes = preprocess_image(payload, cfg)

    assert from_path.shape == (16, 16)
    assert from_path.dtype == np.float64
    assert np.isfinite(from_path).all()
    assert 0.0 <= from_path.min() <= from_path.max() <= 1.0
    np.testing.assert_array_equal(from_path, from_bytes)
    np.testing.assert_array_equal(from_path, preprocess_image(payload, cfg))


def test_center_crop_keeps_the_center_of_a_non_square_image():
    pixels = np.zeros((20, 30), dtype=np.uint8)
    pixels[:, 15:] = 255
    source = Image.fromarray(pixels, mode="L").convert("RGB")

    result = preprocess_image(_encoded_image(source), {"image_size": 20})

    np.testing.assert_array_equal(result[:, :10], 0.0)
    np.testing.assert_array_equal(result[:, 10:], 1.0)


def test_equal_side_input_is_not_resampled():
    pixels = np.array([[0, 64, 128, 192], [255, 192, 128, 64]] * 2, dtype=np.uint8)
    source = Image.fromarray(pixels, mode="L").convert("RGB")

    result = preprocess_image(_encoded_image(source), {"image_size": 4})

    np.testing.assert_array_equal(result, pixels.astype(np.float64) / 255.0)


def test_exif_orientation_is_applied_before_crop():
    pixels = np.zeros((20, 30, 3), dtype=np.uint8)
    pixels[:, :15] = [255, 0, 0]
    pixels[:, 15:] = [0, 0, 255]
    source = Image.fromarray(pixels, mode="RGB")
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    payload = _encoded_image(source, format_name="JPEG", quality=100, exif=exif)

    result = preprocess_image(payload, {"image_size": 20})

    # Orientation 6 moves the original right/blue half to the top.
    assert result[:8].mean() > 0.25
    assert result[-8:].mean() < 0.25


def test_undersized_images_are_rejected_without_upscaling():
    source = _encoded_image(Image.new("RGB", (15, 30), "white"))

    with pytest.raises(ValueError, match="smaller than target"):
        preprocess_image(source, {"image_size": 16})


@pytest.mark.parametrize("payload", [b"not an image", b"\x89PNG\r\n\x1a\ncorrupt"])
def test_corrupt_or_unsupported_bytes_raise_clear_error(payload):
    with pytest.raises(ValueError, match="corrupt or unsupported"):
        preprocess_image(payload, {"image_size": 16})


def test_invalid_source_and_config_are_rejected():
    with pytest.raises(TypeError, match="path or encoded image bytes"):
        preprocess_image(object(), {"image_size": 16})
    with pytest.raises(TypeError, match="cfg must be a mapping"):
        preprocess_image(b"not an image", None)
    with pytest.raises(ValueError, match="positive integer"):
        preprocess_image(b"not an image", {"image_size": 0})
    with pytest.raises(ValueError, match="does not match"):
        preprocess_image(
            b"not an image",
            {"image_size": 16, "preprocess": {"version": 999}},
        )


def test_preprocess_settings_are_versioned_in_project_config_hash():
    cfg = load_config("config.yaml")
    assert cfg["preprocess"] == PREPROCESS_VERSION
    changed = dict(cfg)
    changed["preprocess"] = {**PREPROCESS_VERSION, "version": 2}
    assert config_hash(cfg) != config_hash(changed)

"""Known-answer audit checks on synthetic images and independent metadata."""

from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
import pytest

from aidetect.audit import (PROBE_COLUMNS, audit_table, describe_file, duplicate_groups,
                            estimate_jpeg_quality, normalise_format, shortcut_probe)


@pytest.mark.parametrize("quality", [30, 50, 75, 90, 100])
def test_standard_jpeg_quality_estimate(quality, tmp_path):
    path = tmp_path / "misleading.png"
    Image.new("RGB", (40, 30), "red").save(path, format="JPEG", quality=quality)
    result = describe_file(path)
    assert result["format"] == "JPEG"
    assert result["jpeg_quality"] == quality
    assert result["jpeg_quality_table_mae"] == 0
    assert result["width"] == 40 and result["height"] == 30


def test_png_description_and_corrupt_files(tmp_path):
    path = tmp_path / "wrong.jpg"
    Image.new("RGBA", (32, 48), "blue").save(path, format="PNG")
    result = describe_file(path)
    assert result["format"] == "PNG" and result["jpeg_quality"] is None
    assert result["mode"] == "RGBA" and len(result["phash"]) == 16
    path.write_bytes(b"\x89PNG\r\n\x1a\ncorrupt")
    with pytest.raises(OSError):
        describe_file(path)
    with pytest.raises(ValueError, match="quantization"):
        estimate_jpeg_quality({0: [1]})


def test_custom_quantization_estimate_is_marked_approximate():
    quality, residual = estimate_jpeg_quality({0: list(range(1, 65))})
    assert 1 <= quality <= 100 and residual > 0


def test_normalisation_deterministic_rgb_444_and_original_untouched():
    stream = BytesIO()
    Image.new("RGBA", (40, 30), "red").save(stream, format="PNG")
    original = stream.getvalue()
    encoded = normalise_format(original, 83)
    assert encoded == normalise_format(original, 83)
    assert original.startswith(b"\x89PNG")
    with Image.open(BytesIO(encoded)) as image:
        assert image.format == "JPEG" and image.mode == "RGB"
        assert image.size == (40, 30) and JpegImagePlugin.get_sampling(image) == 0
    with pytest.raises(ValueError):
        normalise_format(original, 0)


def test_probe_near_chance_for_balanced_independent_labels():
    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame({"label": rng.permutation(np.tile([0, 1], n // 2)),
        "is_jpeg": rng.integers(0, 2, n), "jpeg_quality": rng.integers(70, 96, n).astype(float),
        "width": rng.choice([256, 320, 400, 512], n), "height": rng.choice([256, 320, 400, 512], n),
        "n_bytes": rng.integers(3000, 10000, n)})
    df.loc[df["is_jpeg"] == 0, "jpeg_quality"] = np.nan
    df["aspect_ratio"] = df["width"] / df["height"]
    result = shortcut_probe(df, 42)
    assert result["columns"] == PROBE_COLUMNS
    for model in result["models"].values():
        assert .40 <= model["mean_accuracy"] <= .60
        assert len(model["fold_accuracies"]) == 5


def test_probe_detects_format_shortcut_and_rejects_imbalance():
    df = pd.DataFrame({"label": [0] * 50 + [1] * 50,
        "is_jpeg": [1] * 50 + [0] * 50, "jpeg_quality": [85.] * 50 + [np.nan] * 50,
        "width": [256] * 100, "height": [256] * 100, "n_bytes": [1000] * 100,
        "aspect_ratio": [1.] * 100})
    result = shortcut_probe(df, 42)
    assert all(model["mean_accuracy"] == 1 for model in result["models"].values())
    with pytest.raises(ValueError, match="balanced"):
        shortcut_probe(df.iloc[:-1], 42)
    df.loc[0, "width"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        shortcut_probe(df, 42)


def test_audit_table_checks_paths_and_original_hashes(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), "red").save(path)
    digest = describe_file(path)["sha256"]
    manifest = pd.DataFrame([{"id": "sample", "rel_path": "sample.png", "label": 0, "sha256": digest}])
    result = audit_table(manifest, tmp_path)
    assert result.loc[0, "width"] == 32
    manifest.loc[0, "sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed"):
        audit_table(manifest, tmp_path)
    manifest.loc[0, "rel_path"] = "../elsewhere.png"
    with pytest.raises(ValueError, match="escapes"):
        audit_table(manifest, tmp_path)
    with pytest.raises(ValueError, match="unique"):
        audit_table(pd.concat([manifest, manifest]), tmp_path)


def test_duplicate_transitive_union_exact_bytes_and_reorder_stability():
    # A-B and B-C each differ by one bit, A-C by two: union must be transitive.
    frame = pd.DataFrame({"id": ["a", "b", "c", "d", "e"],
        "phash": ["0000", "0001", "0003", "ffff", "abcd"],
        "sha256": ["A", "B", "C", "same", "same"]})
    result = duplicate_groups(frame, 1)
    assert result.iloc[0] == result.iloc[1] == result.iloc[2]
    assert result.iloc[3] == result.iloc[4] != result.iloc[0]
    reordered = frame.sample(frac=1, random_state=3)
    expected = dict(zip(frame["id"], result))
    assert dict(zip(reordered["id"], duplicate_groups(reordered, 1))) == expected


def test_duplicates_preserve_original_relationship_after_normalisation():
    frame = pd.DataFrame({"id": ["a", "b"], "phash": ["0000", "ffff"],
                          "original_phash": ["1234", "1234"]})
    assert duplicate_groups(frame, 0).nunique() == 1

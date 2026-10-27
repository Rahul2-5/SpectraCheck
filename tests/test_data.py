"""Offline acquisition checks using tiny encoded images and real HF features."""

import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from datasets import ClassLabel, Dataset, Features, Image
from PIL import Image as PILImage
import pytest

from aidetect import data


@pytest.fixture
def image_bytes():
    output = {}
    for extension, format_name in (("jpg", "JPEG"), ("png", "PNG")):
        buffer = BytesIO()
        PILImage.new("RGB", (16, 16), "red").save(buffer, format=format_name)
        output[extension] = buffer.getvalue()
    return output


@pytest.fixture
def hub(monkeypatch, image_bytes):
    # SD15 deliberately has index 0, Real index 2: no magic generator IDs.
    features = Features({"image": Image(), "label": ClassLabel(names=["real", "fake"]),
                         "generator": ClassLabel(names=["SD15", "ADM", "Real", "SD14"])})
    rows = {"train": [
        {"image": {"bytes": image_bytes["png"], "path": "misleading.jpg"}, "label": 1, "generator": 0},
        *[{"image": {"bytes": image_bytes["jpg"], "path": "misleading.png"}, "label": 0, "generator": 2} for _ in range(6)],
        {"image": {"bytes": image_bytes["png"], "path": None}, "label": 1, "generator": 1},
    ], "validation": [
        {"image": {"bytes": image_bytes["jpg"], "path": None}, "label": 1, "generator": 0},
        {"image": {"bytes": image_bytes["jpg"], "path": None}, "label": 0, "generator": 2},
    ]}
    calls = []

    def loader(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return Dataset.from_list(rows[kwargs["split"]], features=features).to_iterable_dataset()

    monkeypatch.setattr(data, "load_dataset", loader)
    monkeypatch.setattr(data, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="snapshot-abc")))
    return SimpleNamespace(rows=rows, features=features, calls=calls)


def acquire(tmp_path, **kwargs):
    return data.stream_generator_subset("test/repo", "SD15", ["train", "validation"],
        tmp_path / "main", kwargs.pop("max_real", 3), kwargs.pop("seed", 42), **kwargs)


@pytest.mark.parametrize("extension", ["jpg", "png"])
def test_magic_bytes(image_bytes, extension):
    assert data.detect_image_extension(image_bytes[extension]) == extension


@pytest.mark.parametrize("payload", [b"", b"fake.jpg", b"GIF89a"])
def test_unsupported_magic_rejected(payload):
    with pytest.raises(ValueError, match="signature"):
        data.detect_image_extension(payload)


def test_reservoir_is_bounded_repeatable_and_samples_beyond_prefix():
    first = data.reservoir_sample(iter(range(100)), 10, random.Random(42))
    assert len(first) == len(set(first)) == 10
    assert first == data.reservoir_sample(range(100), 10, random.Random(42))
    assert any(value >= 10 for value in first)
    assert first != data.reservoir_sample(range(100), 10, random.Random(43))
    assert data.reservoir_sample(range(3), 10, random.Random(42)) == [0, 1, 2]
    assert data.reservoir_sample(range(3), 0, random.Random(42)) == []
    with pytest.raises(ValueError):
        data.reservoir_sample([], -1, random.Random())


def test_stream_saves_originals_and_counts_all_generators(tmp_path, hub, image_bytes):
    summary = acquire(tmp_path)
    assert summary["generator_counts"] == {"SD15": 2, "ADM": 1, "Real": 7, "SD14": 0}
    assert summary["saved_counts"] == {"real": 3, "ai": 2}
    assert summary["specification"]["revision"] == "snapshot-abc"
    manifest = tmp_path / "main/manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == data.MANIFEST_COLUMNS
        rows = list(reader)
    assert len(rows) == 5
    assert (tmp_path / "main/ai/train_0.png").read_bytes() == image_bytes["png"]
    for row in rows:
        payload = (tmp_path / "main" / row["rel_path"]).read_bytes()
        assert len(payload) == int(row["n_bytes"])
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert row["config_hash"] == summary["specification"]["config_hash"]
    assert all(call[1]["streaming"] and call[1]["token"] is False for call in hub.calls)
    assert all(call[1]["revision"] == "snapshot-abc" for call in hub.calls)


def test_resume_preserves_files_and_manifest(tmp_path, hub):
    acquire(tmp_path)
    files = list((tmp_path / "main").glob("*/*"))
    modification_times = {path: path.stat().st_mtime_ns for path in files}
    first_manifest = (tmp_path / "main/manifest.csv").read_bytes()
    (tmp_path / "main/ai/train_0.png.tmp").write_bytes(b"interrupted temporary file")
    acquire(tmp_path)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in modification_times.items())
    assert first_manifest == (tmp_path / "main/manifest.csv").read_bytes()


def test_resume_repairs_missing_file_but_rejects_corrupted_existing_file(tmp_path, hub):
    acquire(tmp_path)
    path = tmp_path / "main/ai/train_0.png"
    expected = path.read_bytes()
    path.unlink()
    acquire(tmp_path)
    assert path.read_bytes() == expected
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="differs from source"):
        acquire(tmp_path)


def test_refuses_changed_settings_and_unmanaged_output(tmp_path, hub):
    acquire(tmp_path)
    with pytest.raises(ValueError, match="different acquisition"):
        acquire(tmp_path, seed=43)
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="no acquisition record"):
        data._prepare_output(other, {})
    assert (other / "keep.txt").read_text() == "user data"


def test_refuses_changed_source_revision(tmp_path, hub, monkeypatch):
    acquire(tmp_path)
    monkeypatch.setattr(data, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="new-snapshot")))
    with pytest.raises(ValueError, match="different acquisition"):
        acquire(tmp_path)


def test_refuses_stray_images(tmp_path, hub):
    acquire(tmp_path)
    (tmp_path / "main/real/extra.jpg").write_bytes(b"old unrelated image")
    with pytest.raises(ValueError, match="do not match"):
        acquire(tmp_path)


@pytest.mark.parametrize("names", [[], ["train", "train"], ["../escape"], ["a/b"]])
def test_split_names_cannot_escape_output(names):
    with pytest.raises(ValueError, match="Source split names"):
        data._validate_split_names(names)


def test_rejects_schema_changes(hub):
    with pytest.raises(ValueError, match="must expose"):
        data._check_schema(None, "SD15")
    changed = dict(hub.features)
    changed["generator"] = "plain integer"
    with pytest.raises(ValueError, match="ClassLabel"):
        data._check_schema(changed, "SD15")
    changed = dict(hub.features)
    changed["label"] = ClassLabel(names=["fake", "real"])
    with pytest.raises(ValueError, match="mapping changed"):
        data._check_schema(changed, "SD15")
    with pytest.raises(ValueError):
        data._check_schema(hub.features, "missing-generator")


def test_rejects_label_generator_disagreement(tmp_path, hub):
    hub.rows["train"][0]["label"] = 0
    with pytest.raises(ValueError, match="disagree"):
        acquire(tmp_path)


def test_rejects_empty_target_or_real_class(tmp_path, hub):
    hub.rows["train"] = [row for row in hub.rows["train"] if row["generator"] != 0]
    hub.rows["validation"] = [row for row in hub.rows["validation"] if row["generator"] != 0]
    with pytest.raises(ValueError, match="at least one real"):
        acquire(tmp_path)


def test_original_bytes_local_path_and_failures(tmp_path, image_bytes):
    path = tmp_path / "original.png"
    path.write_bytes(image_bytes["png"])
    assert data._original_bytes({"bytes": None, "path": str(path)}) == image_bytes["png"]
    for bad in (None, {"bytes": None, "path": "missing.jpg"}):
        with pytest.raises(ValueError):
            data._original_bytes(bad)


@pytest.fixture
def cifake(tmp_path, image_bytes):
    root = tmp_path / "cifake"
    for split in ("train", "test"):
        for folder in ("REAL", "FAKE"):
            directory = root / split / folder
            directory.mkdir(parents=True)
            for index in range(4):
                (directory / f"{index}.jpg").write_bytes(image_bytes["png"])
    return root


def test_cifake_balanced_original_byte_pilot_and_resume(tmp_path, cifake, image_bytes):
    output = tmp_path / "pilot"
    result = data.prepare_cifake_sample(cifake, output, 3)
    assert result["saved_counts"] == {"real": 3, "ai": 3}
    assert result["generator_counts"] == {"Real": 8, "SD14": 8}
    images = list(output.glob("*/*.png"))
    assert len(images) == 6
    assert all(path.read_bytes() == image_bytes["png"] for path in images)
    timestamps = [path.stat().st_mtime_ns for path in images]
    data.prepare_cifake_sample(cifake, output, 3)
    assert timestamps == [path.stat().st_mtime_ns for path in images]
    assert json.loads((output / "acquisition.json").read_text())["status"] == "complete"


def test_cifake_rejects_small_missing_or_overlapping_sources(tmp_path, cifake):
    with pytest.raises(ValueError, match="Not enough"):
        data.prepare_cifake_sample(cifake, tmp_path / "pilot", 100)
    with pytest.raises(ValueError, match="must not overlap"):
        data.prepare_cifake_sample(cifake, cifake / "pilot", 1)
    with pytest.raises(ValueError, match="existing extracted"):
        data.prepare_cifake_sample(tmp_path / "missing", tmp_path / "pilot", 1)
    for path in (cifake / "test/REAL").iterdir():
        path.unlink()
    (cifake / "test/REAL").rmdir()
    with pytest.raises(ValueError, match="Missing CIFAKE"):
        data.prepare_cifake_sample(cifake, tmp_path / "pilot", 1)


def test_actual_local_parquet_streaming_round_trip(tmp_path, hub, monkeypatch):
    """Exercise installed datasets' Parquet bytes/features path, offline."""
    from datasets import load_dataset as actual_load_dataset
    paths = {}
    for split, rows in hub.rows.items():
        path = tmp_path / f"{split}.parquet"
        Dataset.from_list(rows, features=hub.features).to_parquet(path)
        paths[split] = str(path)

    # HF repeats its cache path inside lock filenames; use a short temporary
    # workspace directory to stay below Windows' filename-length limit.
    workspace = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="hf-", dir=workspace) as cache:
        assert Path(cache).resolve().parent == workspace

        def local_loader(repo_id, **kwargs):
            return actual_load_dataset("parquet", data_files=paths, split=kwargs["split"],
                                       streaming=True, cache_dir=cache)

        monkeypatch.setattr(data, "load_dataset", local_loader)
        summary = acquire(tmp_path)
        assert summary["saved_counts"] == {"real": 3, "ai": 2}


def test_resume_after_interruption(tmp_path, hub, monkeypatch):
    save_original = data._save_original
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated interrupted session")
        return save_original(*args, **kwargs)

    monkeypatch.setattr(data, "_save_original", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        acquire(tmp_path)
    record = tmp_path / "main/acquisition.json"
    assert json.loads(record.read_text())["status"] == "incomplete"
    existing = {path: path.stat().st_mtime_ns for path in (tmp_path / "main").glob("*/*")}
    monkeypatch.setattr(data, "_save_original", save_original)
    acquire(tmp_path)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in existing.items())
    assert json.loads(record.read_text())["status"] == "complete"

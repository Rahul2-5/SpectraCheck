"""Original-byte acquisition only; no image transformation or train/test split.

Main acquisition scans metadata to select a bounded real reservoir and count
all generators, then streams images to disk. The upstream splits are provenance,
not this project's future train/validation/test assignments.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, TypeVar

from datasets import ClassLabel, Image, load_dataset
from huggingface_hub import HfApi

from aidetect.io_utils import config_hash, sha256_of_file

T = TypeVar("T")
MANIFEST_COLUMNS = [
    "id", "rel_path", "label", "generator", "source_split", "ext", "n_bytes",
    "sha256", "config_hash",
]


def detect_image_extension(payload: bytes) -> str:
    """Identify supported acquisition formats from bytes, never filenames."""
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    raise ValueError("Unsupported image signature: acquisition expects JPEG or PNG")


def reservoir_sample(items: Iterable[T], size: int, rng: random.Random) -> list[T]:
    """Algorithm R: uniform sampling in O(size) memory, with caller-owned RNG."""
    if size < 0:
        raise ValueError("Reservoir size must be non-negative")
    reservoir: list[T] = []
    for index, item in enumerate(items):
        if index < size:
            reservoir.append(item)
        else:
            slot = rng.randrange(index + 1)
            if slot < size:
                reservoir[slot] = item
    return reservoir


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_output(out_dir: Path, specification: dict[str, Any]) -> None:
    """Refuse accidental reuse for different settings; preserve partial runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "acquisition.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous["specification"] != specification:
            raise ValueError("Output belongs to different acquisition settings; choose a new output folder")
    elif any(out_dir.iterdir()):
        raise ValueError("Non-empty output folder has no acquisition record; choose a new folder")
    _atomic_json(state_path, {"specification": specification, "status": "incomplete"})


def _save_original(out_dir: Path, identifier: str, label: int, generator: str,
                   source_split: str, payload: bytes, configuration_hash: str) -> dict[str, Any]:
    extension = detect_image_extension(payload)
    relative = Path("ai" if label == 1 else "real") / f"{identifier}.{extension}"
    destination = out_dir / relative
    expected_hash = hashlib.sha256(payload).hexdigest()
    if destination.exists():
        if sha256_of_file(destination) != expected_hash:
            raise ValueError(f"Existing image differs from source bytes: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return dict(zip(MANIFEST_COLUMNS, [identifier, relative.as_posix(), label,
        generator, source_split, extension, len(payload), expected_hash, configuration_hash]))


def _finish_output(out_dir: Path, rows: list[dict[str, Any]], specification: dict[str, Any],
                   counts: dict[str, int], schemas: dict[str, Any]) -> dict[str, Any]:
    expected = {row["rel_path"] for row in rows}
    actual = {path.relative_to(out_dir).as_posix() for folder in ("real", "ai")
              for path in (out_dir / folder).glob("*") if path.is_file() and path.suffix != ".tmp"}
    if len(expected) != len(rows) or actual != expected:
        raise ValueError("Image files do not match the selected manifest; inspect the output folder")
    manifest = out_dir / "manifest.csv"
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    summary = {"specification": specification, "status": "complete",
               "generator_counts": counts, "schemas": schemas,
               "saved_counts": {"real": sum(row["label"] == 0 for row in rows),
                                "ai": sum(row["label"] == 1 for row in rows)},
               "manifest_sha256": sha256_of_file(manifest)}
    _atomic_json(out_dir / "acquisition.json", summary)
    print("Source generator counts:", json.dumps(counts, sort_keys=True))
    print("Saved images:", json.dumps(summary["saved_counts"], sort_keys=True))
    return summary


def _validate_split_names(split_names: Iterable[str]) -> list[str]:
    names = list(split_names)
    if not names or len(set(names)) != len(names) or any(
        not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in names
    ):
        raise ValueError("Source split names must be unique non-empty names without path separators")
    return names


def _check_schema(features: Any, generator_name: str) -> dict[str, Any]:
    if features is None or not all(key in features for key in ("image", "label", "generator")):
        raise ValueError("Dataset must expose image, label, and generator features")
    if not isinstance(features["image"], Image) or not all(
        isinstance(features[name], ClassLabel) for name in ("label", "generator")
    ):
        raise ValueError("Expected Image and ClassLabel features; inspect the dataset schema")
    labels = features["label"]
    if labels.names != ["real", "fake"]:
        raise ValueError("Expected binary labels real=0 and fake=1; dataset mapping changed")
    generators = features["generator"]
    # The target and Real IDs are looked up, even when the target is not ID 6.
    return {"labels": labels.names, "generators": generators.names,
            "target_id": generators.str2int(generator_name), "real_id": generators.str2int("Real")}


def _original_bytes(image: Any) -> bytes:
    if not isinstance(image, dict):
        raise ValueError("Expected undecoded image dictionary; use Image(decode=False)")
    if image.get("bytes") is not None:
        return bytes(image["bytes"])
    if image.get("path") and Path(image["path"]).is_file():
        return Path(image["path"]).read_bytes()
    raise ValueError("Image has neither embedded bytes nor an accessible local source file")


def stream_generator_subset(repo_id: str, generator_name: str, split_names: Iterable[str],
                            out_dir: str | Path, max_real: int, seed: int, *,
                            revision: str = "main", configuration_hash: str | None = None) -> dict[str, Any]:
    """Save all target-generator rows and a uniform capped sample of real rows.

    Resolve the Hub revision to a commit before either pass, disabling implicit
    credentials. Reruns rescan the same pinned snapshot but never rewrite valid
    saved images. Resume saves disk work, not a network-stream offset. The full
    upstream Parquet data may still be transferred; no download-size guarantee.
    """
    names = _validate_split_names(split_names)
    if max_real < 1:
        raise ValueError("max_real must be positive")
    commit = HfApi().dataset_info(repo_id, revision=revision, token=False).sha
    if not commit:
        raise ValueError("Hub did not provide a dataset commit hash")
    specification = {"kind": "main", "repo_id": repo_id, "revision": commit,
                     "generator_name": generator_name, "source_splits": names,
                     "max_real": max_real, "seed": seed}
    digest = configuration_hash or config_hash(specification)
    specification["config_hash"] = digest
    output = Path(out_dir)
    _prepare_output(output, specification)
    streams, schemas, targets = {}, {}, set()
    counts: Counter[str] = Counter()

    # Only metadata references, never thousands of decoded images, enter RAM.
    def real_references():
        for split in names:
            print(f"Scanning source metadata: {split}", flush=True)
            dataset = load_dataset(repo_id, split=split, streaming=True, revision=commit, token=False)
            schema = _check_schema(dataset.features, generator_name)
            schemas[split] = schema
            streams[split] = dataset.cast_column("image", Image(decode=False))
            counts.update({name: 0 for name in schema["generators"]})
            for index, row in enumerate(streams[split].select_columns(["label", "generator"])):
                generator_id = row["generator"]
                if not isinstance(generator_id, int) or not 0 <= generator_id < len(schema["generators"]):
                    raise ValueError("Invalid generator value in source row")
                generator = schema["generators"][generator_id]
                counts[generator] += 1
                label = row["label"]
                if label not in (0, 1) or (label == 0) != (generator_id == schema["real_id"]):
                    raise ValueError("Binary label and generator disagree in source row")
                if generator_id == schema["target_id"]:
                    targets.add((split, index))
                elif label == 0:
                    yield (split, index)

    selected_real = set(reservoir_sample(real_references(), max_real, random.Random(seed)))
    if not targets or not selected_real:
        raise ValueError("Acquisition needs at least one real and one target-generator image")
    selected = targets | selected_real
    rows = []
    for split in names:
        print(f"Saving selected original images: {split}", flush=True)
        schema = schemas[split]
        for index, row in enumerate(streams[split]):
            if (split, index) in selected:
                generator = schema["generators"][row["generator"]]
                rows.append(_save_original(output, f"{split}_{index}", row["label"], generator,
                            split, _original_bytes(row["image"]), digest))
    if len(rows) != len(selected):
        raise ValueError("Streaming row count changed between metadata and image passes")
    return _finish_output(output, rows, specification, dict(counts), schemas)


def prepare_cifake_sample(source_dir: str | Path, out_dir: str | Path,
                         samples_per_class: int = 500, seed: int = 42, *,
                         split_names: Iterable[str] = ("train", "test"),
                         configuration_hash: str | None = None) -> dict[str, Any]:
    """Copy a pilot from an already extracted Kaggle CIFAKE train/test folder.

    Expected layout: source/{train,test}/{REAL,FAKE}/images. Source split names
    are provenance only. No credentials, download, re-encoding, or upscaling.
    """
    source, output = Path(source_dir).resolve(), Path(out_dir).resolve()
    names = _validate_split_names(split_names)
    if not source.is_dir() or samples_per_class < 1:
        raise ValueError("Provide an existing extracted CIFAKE folder and positive sample size")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Pilot source and output folders must not overlap")
    candidates: dict[int, list[tuple[str, Path]]] = {0: [], 1: []}
    for split in names:
        split_dir = source / split
        if not split_dir.exists():
            continue
        for label, folder in ((0, "REAL"), (1, "FAKE")):
            class_dir = split_dir / folder
            if not class_dir.is_dir():
                raise ValueError(f"Missing CIFAKE class folder: {class_dir}")
            candidates[label].extend((split, path) for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if any(len(paths) < samples_per_class for paths in candidates.values()):
        raise ValueError("Not enough CIFAKE images in both classes for the requested pilot")
    specification = {"kind": "pilot", "source_dir": str(source), "source_splits": names,
                     "samples_per_class": samples_per_class, "seed": seed}
    digest = configuration_hash or config_hash(specification)
    specification["config_hash"] = digest
    _prepare_output(output, specification)
    rng = random.Random(seed)
    rows = []
    for label, paths in candidates.items():
        selected = sorted(reservoir_sample(paths, samples_per_class, rng))
        for index, (split, path) in enumerate(selected):
            rows.append(_save_original(output, f"{split}_{label}_{index}", label,
                "SD14" if label else "Real", split, path.read_bytes(), digest))
    return _finish_output(output, rows, specification,
        {"Real": len(candidates[0]), "SD14": len(candidates[1])},
        {"layout": "source/{train,test}/{REAL,FAKE}"})

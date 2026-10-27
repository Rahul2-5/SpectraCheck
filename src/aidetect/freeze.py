"""Create and verify the immutable Phase 7 evaluation manifest."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from aidetect.io_utils import config_hash, sha256_of_file


_MODEL_KEYS = tuple(
    f"{feature_set}_{model}"
    for feature_set in ("frequency", "pixel")
    for model in ("svm", "rf", "xgb")
)
_TARGET_FIELDS = {
    "status": re.compile(r"^Status:\s*(.+?)\s*$", re.MULTILINE),
    "minimum_validation_accuracy": re.compile(
        r"^Minimum validation accuracy:\s*`?([0-9]+(?:\.[0-9]+)?)`?\s*$", re.MULTILINE
    ),
    "rationale": re.compile(r"^Rationale:\s*(.+?)\s*$", re.MULTILINE),
    "agreed_by": re.compile(r"^Agreed by:\s*(.+?)\s*$", re.MULTILINE),
    "supervisor": re.compile(r"^Supervisor:\s*(.+?)\s*$", re.MULTILINE),
    "decision_date": re.compile(r"^Decision date:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE),
}


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, check=False, text=True, capture_output=True
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _repo_root(path: str | Path | None = None) -> Path:
    requested = Path(path).resolve() if path is not None else Path(__file__).resolve().parents[2]
    top = Path(_git(requested, "rev-parse", "--show-toplevel")).resolve()
    if requested != top and top not in requested.parents:
        raise ValueError("The requested repository root is not inside the Git repository")
    return top


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative(repo_root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository for a reproducible freeze") from exc


def _assert_main_clean(repo_root: Path) -> tuple[str, str]:
    branch = _git(repo_root, "branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"Phase 7 freeze must run on main; current branch is {branch!r}")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "Phase 7 freeze requires a clean Git tree; commit or remove working-tree changes first"
        )
    commit = _git(repo_root, "rev-parse", "HEAD")
    return branch, commit


def _tracked_python_sources(repo_root: Path) -> list[Path]:
    output = _git(repo_root, "ls-files", "-z", "--", "src/aidetect")
    files = [repo_root / item for item in output.split("\0") if item.endswith(".py")]
    files = sorted(path for path in files if path.is_file())
    if not files:
        raise FileNotFoundError("No tracked Python sources found under src/aidetect")
    return files


def _tracked_python_scripts(repo_root: Path) -> list[Path]:
    output = _git(repo_root, "ls-files", "-z", "--", "scripts")
    files = [repo_root / item for item in output.split("\0") if item.endswith(".py")]
    files = sorted(path for path in files if path.is_file())
    if not files:
        raise FileNotFoundError("No tracked Python scripts found under scripts/")
    return files


def validate_practical_target(path: str | Path) -> dict[str, Any]:
    """Require the team's pre-test decision and reject the untouched template."""
    target_path = Path(path)
    content = target_path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for name, pattern in _TARGET_FIELDS.items():
        matches = pattern.findall(content)
        if len(matches) != 1:
            raise ValueError(
                f"{target_path} must contain exactly one completed {name.replace('_', ' ')} field"
            )
        values[name] = matches[0].strip()
    if values["status"].lower() != "agreed":
        raise ValueError("Practical-target Status must be exactly 'Agreed' before the freeze")
    threshold = float(values["minimum_validation_accuracy"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Minimum validation accuracy must be between 0 and 1")
    for field in ("rationale", "agreed_by", "supervisor"):
        text = values[field]
        if not text or re.search(r"pending|todo|placeholder|\[|\]|<|>", text, re.IGNORECASE):
            raise ValueError(f"The practical-target {field.replace('_', ' ')} is still a placeholder")
    try:
        decision_date = date.fromisoformat(values["decision_date"])
    except ValueError as exc:
        raise ValueError("Decision date must be a valid ISO date (YYYY-MM-DD)") from exc
    return {
        "minimum_validation_accuracy": threshold,
        "rationale": values["rationale"],
        "agreed_by": values["agreed_by"],
        "supervisor": values["supervisor"],
        "decision_date": decision_date.isoformat(),
    }


def _validate_split_file(path: Path, expected_digest: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Immutable split file is missing: {path}")
    actual_digest = sha256_of_file(path)
    if actual_digest != expected_digest:
        raise ValueError("splits.csv does not match artifacts/splits.sha256")
    frame = pd.read_csv(path, dtype={"id": str, "group_id": str, "split": str})
    required = {"id", "label", "split", "group_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"splits.csv is missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("splits.csv must contain non-empty IDs, labels, split names, and groups")
    if frame["id"].astype(str).str.strip().eq("").any() or frame["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("splits.csv IDs and group IDs must not be blank")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("splits.csv contains duplicate IDs")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("splits.csv labels must be 0=real and 1=AI")
    frame["label"] = labels.astype("int64")
    if set(frame["split"].astype(str).unique()) != {"train", "val", "test"}:
        raise ValueError("splits.csv must contain train, val, and test assignments")
    group_splits = frame.groupby(frame["group_id"].astype(str))["split"].nunique()
    if (group_splits > 1).any():
        raise ValueError("A duplicate group crosses frozen splits")
    counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    if set(counts.columns) != {0, 1} or (counts[0] != counts[1]).any():
        raise ValueError("Every frozen split must contain exactly balanced real and AI classes")
    return {
        "rows": int(len(frame)),
        "counts": {
            split: {str(int(label)): int(count) for label, count in row.items()}
            for split, row in counts.iterrows()
        },
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Required {label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Required {label} must contain a JSON object")
    return result


def _validate_sealed_batch_metadata(
    batches: list[Path],
    *,
    cfg: dict[str, Any],
    active_config_digest: str,
    splits_digest: str,
    expected_rows: int,
) -> None:
    """Check only Parquet footers/schema before lock; never read test rows."""
    image_size = int(cfg.get("image_size", 0))
    rings = int(cfg.get("rapsd_rings", 0))
    pixel_size = int(cfg.get("pixel_baseline_size", 0))
    thresholds = cfg.get("extraction", {}).get("peak_k_mad", [])
    if min(image_size, rings, pixel_size) < 1:
        raise ValueError("Image size, RAPSD rings, and pixel baseline size must be positive")
    feature_contract = {
        name: cfg.get(name)
        for name in (
            "image_size",
            "rapsd_rings",
            "pixel_baseline_size",
            "preprocess",
            "frequency",
            "extraction",
        )
    }
    training_cfg = cfg.get("training", {})
    expected_contract = (
        training_cfg.get("phase5_feature_contract_sha256")
        if isinstance(training_cfg, dict)
        else None
    )
    legacy_hashes = set(training_cfg.get("legacy_phase5_config_hashes", [])) if isinstance(training_cfg, dict) else set()
    contract_matches = config_hash(feature_contract) == expected_contract
    expected_metadata = {
        b"aidetect_schema_version": b"1",
        b"aidetect_splits_sha256": splits_digest.encode("ascii"),
        b"aidetect_split": b"test",
        b"aidetect_image_side": str(image_size).encode("ascii"),
        b"aidetect_rapsd_rings": str(rings).encode("ascii"),
        b"aidetect_pixel_baseline_size": str(pixel_size).encode("ascii"),
        b"aidetect_peak_k_mad": json.dumps(list(thresholds)).encode("ascii"),
    }
    required_columns = {
        "id",
        "label",
        "group_id",
        "split",
        "entropy",
        *[f"cumulative_energy_{index:03d}" for index in range(math.ceil(math.sqrt(2) * (image_size // 2)) + 1)],
        *[f"rapsd_ring_{index:02d}" for index in range(rings)],
        *[f"pixel_{index:04d}" for index in range(pixel_size * pixel_size)],
        *[
            column
            for threshold in thresholds
            for column in (f"peak_count_k{float(threshold):g}", f"peak_strength_k{float(threshold):g}")
        ],
    }
    total_rows = 0
    for index, path in enumerate(batches):
        metadata = pq.read_metadata(path)
        raw_metadata = metadata.metadata or {}
        mismatches = [
            key.decode("ascii")
            for key, value in expected_metadata.items()
            if raw_metadata.get(key) != value
        ]
        stored_config_hash = raw_metadata.get(b"aidetect_config_hash", b"").decode("ascii").lower()
        if stored_config_hash != active_config_digest and stored_config_hash not in legacy_hashes:
            mismatches.append("aidetect_config_hash")
        if stored_config_hash in legacy_hashes and not contract_matches:
            raise ValueError(
                "Legacy Phase 5 sealed features are compatible only while the recorded "
                "preprocessing/frequency feature contract remains unchanged"
            )
        expected_index = str(index).encode("ascii")
        if raw_metadata.get(b"aidetect_batch_index") != expected_index:
            mismatches.append("aidetect_batch_index")
        ids_digest = raw_metadata.get(b"aidetect_batch_ids_sha256", b"").decode("ascii").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", ids_digest):
            mismatches.append("aidetect_batch_ids_sha256")
        missing_columns = sorted(required_columns - set(metadata.schema.names))
        if missing_columns:
            mismatches.append("feature_columns:" + ",".join(missing_columns))
        if mismatches:
            raise ValueError(
                f"Sealed feature batch {path} has incompatible footer metadata/schema: "
                + ", ".join(mismatches)
            )
        total_rows += int(metadata.num_rows)
    if total_rows != expected_rows:
        raise ValueError(
            f"Sealed test batch metadata row count ({total_rows}) does not match the frozen split ({expected_rows})"
        )


def _sha_file_map(repo_root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        _relative(repo_root, path, "A frozen file"): sha256_of_file(path)
        for path in sorted(paths, key=lambda item: item.as_posix())
    }


def _collect_inputs(
    cfg: dict[str, Any], *, config_file: str | Path, repo_root: Path
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, Any]]:
    config_path = _resolve(repo_root, config_file)
    _relative(repo_root, config_path, "config.yaml")
    paths_cfg = cfg.get("paths", {})
    if not isinstance(paths_cfg, dict):
        raise ValueError("config.yaml paths must be a mapping")
    artifacts_root = _resolve(repo_root, paths_cfg.get("artifacts_root", "artifacts"))
    reports_root = _resolve(repo_root, paths_cfg.get("reports_root", "reports"))
    data_root = _resolve(repo_root, paths_cfg.get("data_root", "data"))
    features_root = _resolve(repo_root, paths_cfg.get("features_root", "features"))
    _relative(repo_root, artifacts_root, "artifacts_root")
    _relative(repo_root, reports_root, "reports_root")

    split_path = data_root / "splits.csv"
    split_digest_path = artifacts_root / "splits.sha256"
    if not split_digest_path.is_file():
        raise FileNotFoundError(f"Immutable split digest is missing: {split_digest_path}")
    recorded_digest = split_digest_path.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_digest):
        raise ValueError("artifacts/splits.sha256 must contain one lowercase SHA-256 digest")
    split_info = _validate_split_file(split_path, recorded_digest)

    training_outputs = {
        "feature_choice.json": _read_json(artifacts_root / "feature_choice.json", "feature choice"),
        "primary_model.json": _read_json(artifacts_root / "primary_model.json", "primary model"),
        "versions.json": _read_json(artifacts_root / "versions.json", "Phase 6 versions"),
        "training_results.json": _read_json(artifacts_root / "training_results.json", "Phase 6 results"),
    }
    active_config_digest = config_hash(cfg)
    for name in ("feature_choice.json", "primary_model.json", "training_results.json"):
        record = training_outputs[name]
        if record.get("config_sha256") != active_config_digest:
            raise ValueError(f"Phase 6 {name} does not match the active config hash")
        if record.get("splits_sha256") != recorded_digest:
            raise ValueError(f"Phase 6 {name} does not match the immutable split hash")
    if training_outputs["training_results.json"].get("test_features_read") is not False:
        raise ValueError("Phase 6 provenance must explicitly confirm test_features_read is false")
    if training_outputs["primary_model.json"].get("feature_set") != "frequency":
        raise ValueError("The validation-selected primary model must use frequency features")
    primary_key = training_outputs["primary_model.json"].get("artifact_key")
    if primary_key not in {f"frequency_{name}" for name in ("svm", "rf", "xgb")}:
        raise ValueError("primary_model.json must select one configured frequency model")
    choice = training_outputs["feature_choice.json"]
    training_cfg = cfg.get("training", {})
    r0_cfg = training_cfg.get("r0", {}) if isinstance(training_cfg, dict) else {}
    r0_min, r0_max, r0_step = (int(r0_cfg.get(name, 0)) for name in ("min", "max", "step"))
    if r0_min < 1 or r0_max < r0_min or r0_step < 1:
        raise ValueError("Configured training.r0 grid is invalid")
    if int(choice.get("r0", -1)) not in range(r0_min, r0_max + 1, r0_step):
        raise ValueError("feature_choice.json contains an r0 outside the configured Phase 6 grid")
    if float(choice.get("k_mad", -1)) not in [float(value) for value in cfg.get("extraction", {}).get("peak_k_mad", [])]:
        raise ValueError("feature_choice.json contains a k_mad outside the configured grid")
    model_records = training_outputs["training_results.json"].get("models", [])
    recorded_model_keys = {item.get("key") for item in model_records if isinstance(item, dict)}
    if len(model_records) != len(_MODEL_KEYS) or recorded_model_keys != set(_MODEL_KEYS):
        raise ValueError("Phase 6 results must record the six configured model pipelines")
    if training_outputs["training_results.json"].get("primary_model", {}).get("artifact_key") != primary_key:
        raise ValueError("Phase 6 run record and primary_model.json disagree on the primary model")
    if training_outputs["primary_model.json"].get("selection_metric") != "validation_accuracy":
        raise ValueError("The primary model must have been selected by validation accuracy")

    target_path = reports_root / "practical_target.md"
    practical_target = validate_practical_target(target_path)
    package_sources = _tracked_python_sources(repo_root)
    model_paths = [artifacts_root / "models" / f"{key}.joblib" for key in _MODEL_KEYS]
    for model_path in model_paths:
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise FileNotFoundError(f"Required Phase 6 model artifact is missing or empty: {model_path}")

    tracked_paths = [
        config_path,
        split_digest_path,
        artifacts_root / "feature_choice.json",
        artifacts_root / "primary_model.json",
        artifacts_root / "versions.json",
        artifacts_root / "training_results.json",
        target_path,
        *model_paths,
        *package_sources,
        *_tracked_python_scripts(repo_root),
    ]
    for path in tracked_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required freeze input is missing: {path}")
    hashes = _sha_file_map(repo_root, tracked_paths)
    hashes["external:data_root/splits.csv"] = sha256_of_file(split_path)
    if hashes["external:data_root/splits.csv"] != recorded_digest:
        raise AssertionError("Internal split hash validation failed")

    test_feature_dir = features_root / "test_sealed"
    test_batches = sorted(test_feature_dir.glob("batch_*.parquet"))
    if not test_batches:
        raise FileNotFoundError(f"No sealed test feature batches found under {test_feature_dir}")
    invalid_batches = [path.name for path in test_batches if path.is_symlink() or not path.is_file()]
    if invalid_batches:
        raise ValueError("Sealed feature batches must be regular files: " + ", ".join(invalid_batches))
    unexpected_entries = [
        path.name
        for path in test_feature_dir.iterdir()
        if path not in test_batches
    ]
    if unexpected_entries:
        raise ValueError(
            "Sealed test feature directory contains unexpected files: "
            + ", ".join(sorted(unexpected_entries))
        )
    expected_test_rows = int(split_info["counts"]["test"]["0"]) + int(
        split_info["counts"]["test"]["1"]
    )
    _validate_sealed_batch_metadata(
        test_batches,
        cfg=cfg,
        active_config_digest=active_config_digest,
        splits_digest=recorded_digest,
        expected_rows=expected_test_rows,
    )
    sealed_batches = [
        {"path": batch.relative_to(features_root).as_posix(), "sha256": sha256_of_file(batch)}
        for batch in test_batches
    ]
    return hashes, sealed_batches, {
        "config_sha256": active_config_digest,
        "splits_sha256": recorded_digest,
        "split_summary": split_info,
        "expected_test_rows": expected_test_rows,
        "primary_model": primary_key,
        "practical_target": practical_target,
        "artifacts_root": artifacts_root,
        "reports_root": reports_root,
        "features_root": features_root,
        "data_root": data_root,
        "split_path": split_path,
        "lock_path": artifacts_root / "FREEZE.lock",
    }


def freeze_phase7(
    cfg: dict[str, Any],
    *,
    config_file: str | Path = "config.yaml",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate Phase 6 outputs and create a one-time freeze manifest.

    Hashing sealed test batches reads their raw bytes for integrity only. No
    Parquet table, label, row ID, or feature value is parsed by this function.
    """
    root = _repo_root(repo_root)
    paths_cfg = cfg.get("paths", {})
    if isinstance(paths_cfg, dict):
        existing_lock = _resolve(root, paths_cfg.get("artifacts_root", "artifacts")) / "FREEZE.lock"
        if existing_lock.exists():
            raise FileExistsError(f"{existing_lock} already exists; Phase 7 freeze is single-use")
    branch, commit = _assert_main_clean(root)
    hashes, sealed_batches, details = _collect_inputs(cfg, config_file=config_file, repo_root=root)
    lock_path: Path = details["lock_path"]
    if lock_path.exists():
        raise FileExistsError(f"{lock_path} already exists; Phase 7 freeze is single-use")
    stale_evaluation_outputs = [
        path
        for path in (
            details["artifacts_root"] / "EVALUATION_STARTED.json",
            details["reports_root"] / "results.json",
            details["reports_root"] / "results.md",
            details["reports_root"] / "test_predictions.csv",
        )
        if path.exists()
    ]
    if stale_evaluation_outputs:
        raise FileExistsError(
            "Phase 7 evaluation artifacts already exist; do not reset the one-time guard: "
            + ", ".join(str(path) for path in stale_evaluation_outputs)
        )
    lock = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_branch": branch,
        "git_commit": commit,
        "config_sha256": details["config_sha256"],
        "splits_sha256": details["splits_sha256"],
        "primary_model": details["primary_model"],
        "practical_target": details["practical_target"],
        "expected_test_rows": details["expected_test_rows"],
        "hashes": hashes,
        "sealed_test_batches": sealed_batches,
        "test_feature_values_read_during_freeze": False,
    }
    payload = (json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Do not remove a partial lock: a failed freeze must fail closed.
        raise
    return lock


def verify_freeze_lock(
    cfg: dict[str, Any],
    *,
    config_file: str | Path = "config.yaml",
    repo_root: str | Path | None = None,
    require_committed_lock: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Verify every frozen project and sealed-batch hash before evaluation."""
    root = _repo_root(repo_root)
    branch, current_commit = _assert_main_clean(root)
    hashes, sealed_batches, details = _collect_inputs(cfg, config_file=config_file, repo_root=root)
    lock_path: Path = details["lock_path"]
    if not lock_path.is_file():
        raise FileNotFoundError(f"Phase 7 freeze lock is missing: {lock_path}")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("FREEZE.lock is not valid UTF-8 JSON") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("FREEZE.lock has an unsupported or invalid schema")
    if lock.get("git_branch") != "main" or branch != "main":
        raise ValueError("FREEZE.lock and evaluation must both use main")
    if lock.get("hashes") != hashes:
        expected = lock.get("hashes", {})
        changed = sorted(set(expected) | set(hashes))
        changed = [key for key in changed if expected.get(key) != hashes.get(key)]
        raise ValueError("Freeze input hash mismatch: " + ", ".join(changed))
    if lock.get("sealed_test_batches") != sealed_batches:
        raise ValueError("Sealed test feature batch list or file hash differs from FREEZE.lock")
    for key, expected in (
        ("config_sha256", details["config_sha256"]),
        ("splits_sha256", details["splits_sha256"]),
        ("primary_model", details["primary_model"]),
        ("expected_test_rows", details["expected_test_rows"]),
    ):
        if lock.get(key) != expected:
            raise ValueError(f"FREEZE.lock {key} does not match the verified project inputs")
    frozen_commit = lock.get("git_commit")
    if not isinstance(frozen_commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", frozen_commit):
        raise ValueError("FREEZE.lock has an invalid source commit")
    _git(root, "merge-base", "--is-ancestor", frozen_commit, current_commit)
    if require_committed_lock:
        relative_lock = _relative(root, lock_path, "FREEZE.lock")
        _git(root, "ls-files", "--error-unmatch", "--", relative_lock)
    return lock, root


__all__ = ["freeze_phase7", "validate_practical_target", "verify_freeze_lock"]

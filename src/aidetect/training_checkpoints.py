"""Committed, hash-verified Phase 6 checkpoints; never inspect test features."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import joblib

from aidetect.io_utils import config_hash, sha256_of_file


CHECKPOINT_SCHEMA_VERSION = 1
_KEYS = {"feature_choice", *(
    f"{feature}_{model}" for feature in ("frequency", "pixel")
    for model in ("svm", "rf", "xgb")
)}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if sha256_of_file(source) != sha256_of_file(temporary):
            raise ValueError(f"Checkpoint copy verification failed: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingCheckpoints:
    """One writer, atomic receipt commits, and resumable final publication.

    A killed runtime can leave ``running.lock``. It must be removed only after
    the previous writer is confirmed stopped; a new runtime cannot infer that.
    The manifest commits receipts only after model and receipt writes finish.
    Orphan files from an uncommitted write are not loaded as completed work.
    """

    def __init__(self, root: str | Path, contract: Mapping[str, Any]):
        self.root = Path(root)
        self.contract = dict(contract)
        self.digest = config_hash(self.contract)
        self.manifest: dict[str, Any] = {}
        self.active_model: str | None = None
        self.activity_started = False

    def __enter__(self) -> "TrainingCheckpoints":
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / "running.lock"
        try:
            with self.lock.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_utc": self._now()}, stream)
        except FileExistsError as error:
            raise RuntimeError(
                f"Checkpoint writer lock exists: {self.lock}. Do not start concurrently. "
                "After confirming the previous runtime/process stopped, remove only this stale lock."
            ) from error
        try:
            path = self.root / "manifest.json"
            if path.exists():
                self.manifest = json.loads(path.read_text(encoding="utf-8"))
                if (self.manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                    or self.manifest.get("contract_sha256") != self.digest
                    or self.manifest.get("contract") != self.contract):
                    raise ValueError("Checkpoint contract differs: config, inputs, folds, code or versions changed")
                if not set(self.manifest.get("entries", {})) <= _KEYS:
                    raise ValueError("Checkpoint manifest has unknown entries")
                for key in self.manifest["entries"]:
                    self._receipt(key)  # Verify all committed bytes before unpickling or fitting.
                for name in self.manifest.get("search_entries", {}):
                    self._candidate_receipt(name)
            else:
                unexpected = [p for p in self.root.iterdir()
                              if p.name not in {"running.lock", ".manifest.json.tmp"}]
                if unexpected:
                    raise ValueError("Checkpoint folder has files but no committed manifest")
                self.manifest = {"schema_version": CHECKPOINT_SCHEMA_VERSION,
                                 "contract_sha256": self.digest, "contract": self.contract,
                                 "entries": {}, "published": {}, "complete": False}
                self._commit()
            return self
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc is not None and self.activity_started:
                self.progress("interrupted", self.active_model)
        finally:
            self.lock.unlink(missing_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _commit(self) -> None:
        _atomic_json(self.root / "manifest.json", self.manifest)

    def _receipt(self, key: str) -> dict[str, Any] | None:
        if key not in _KEYS:
            raise ValueError(f"Unknown checkpoint key: {key}")
        digest = self.manifest["entries"].get(key)
        if digest is None:
            return None
        path = self.root / f"{key}.json"
        if not path.is_file() or sha256_of_file(path) != digest:
            raise ValueError(f"Checkpoint receipt checksum mismatch: {key}")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("key") != key or receipt.get("contract_sha256") != self.digest:
            raise ValueError(f"Checkpoint receipt provenance mismatch: {key}")
        if key != "feature_choice":
            model = self.root / f"{key}.joblib"
            if not model.is_file() or sha256_of_file(model) != receipt.get("model_sha256"):
                raise ValueError(f"Checkpoint model checksum mismatch: {key}")
        return receipt

    def load(self, key: str) -> tuple[dict[str, Any], Any] | None:
        receipt = self._receipt(key)
        if receipt is None:
            return None
        model = None if key == "feature_choice" else joblib.load(self.root / f"{key}.joblib")
        return receipt["payload"], model

    def save(self, key: str, payload: Mapping[str, Any], model: Any = None) -> None:
        if key not in _KEYS or key in self.manifest["entries"]:
            raise ValueError(f"Invalid or already committed checkpoint: {key}")
        receipt = {"key": key, "contract_sha256": self.digest, "payload": dict(payload)}
        if key != "feature_choice":
            target = self.root / f"{key}.joblib"
            temporary = target.with_name(f".{target.name}.tmp")
            try:
                joblib.dump(model, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            receipt["model_sha256"] = sha256_of_file(target)
        path = self.root / f"{key}.json"
        _atomic_json(path, receipt)
        self.manifest["entries"][key] = sha256_of_file(path)
        self._commit()

    def _candidate_receipt(self, name: str) -> dict[str, Any] | None:
        import re
        if not re.fullmatch(r"(?:frequency|pixel)_(?:svm|rf|xgb)_\d{4}", name):
            raise ValueError("Invalid candidate checkpoint name")
        digest = self.manifest.get("search_entries", {}).get(name)
        if digest is None:
            return None
        path = self.root / "search" / f"{name}.json"
        if not path.is_file() or sha256_of_file(path) != digest:
            raise ValueError(f"Search candidate checksum mismatch: {name}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("contract_sha256") != self.digest or result.get("name") != name:
            raise ValueError("Search candidate contract mismatch")
        return result["payload"]

    def load_candidate(self, key: str, index: int) -> dict[str, Any] | None:
        return self._candidate_receipt(f"{key}_{index:04d}")

    def save_candidate(self, key: str, index: int, payload: Mapping[str, Any]) -> None:
        name = f"{key}_{index:04d}"
        if self._candidate_receipt(name) is not None:
            raise ValueError("Search candidate already committed")
        path = self.root / "search" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        _atomic_json(path, {"name": name, "contract_sha256": self.digest, "payload": dict(payload)})
        self.manifest.setdefault("search_entries", {})[name] = sha256_of_file(path)
        self._commit()

    def progress(self, stage: str, model: str | None = None) -> None:
        self.activity_started = True
        self.active_model = model
        completed = sorted(set(self.manifest["entries"]) - {"feature_choice"})
        status = {"stage": stage, "current_model": model, "completed_models": completed,
                  "updated_utc": self._now(), "contract_sha256": self.digest,
                  "test_features_read": False}
        _atomic_json(self.root / "status.json", status)
        print(f"[Phase 6] {stage}: {model or 'feature selection/finalization'}; "
              f"{len(completed)}/6 models checkpointed", file=sys.stderr, flush=True)

    def validate_outputs(self, paths: list[Path]) -> None:
        for path in paths:
            if path.exists():
                expected = self.manifest["published"].get(str(path.resolve()))
                if expected is None or sha256_of_file(path) != expected:
                    raise FileExistsError(f"Phase 6 output is not owned by this checkpoint or changed: {path}")
        if self.manifest["complete"] and all(p.is_file() for p in paths):
            raise FileExistsError("Phase 6 outputs already exist and are complete; do not rerun")

    def publish(self, source: Path, target: Path) -> None:
        digest = sha256_of_file(source)
        name = str(target.resolve())
        previous = self.manifest["published"].get(name)
        if previous is not None and previous != digest:
            raise ValueError(f"Resumed final output differs from the committed publication: {target}")
        if target.exists():
            if previous != digest or sha256_of_file(target) != digest:
                raise FileExistsError(f"Refusing to overwrite changed/unowned Phase 6 output: {target}")
            return
        # Reserve exact bytes first. A crash between reservation and installation is resumable.
        self.manifest["published"][name] = digest
        self._commit()
        _atomic_copy(source, target)

    def finish(self) -> None:
        if set(self.manifest["entries"]) != _KEYS:
            raise RuntimeError("Cannot finish an incomplete checkpoint run")
        self.manifest["complete"] = True
        self._commit()
        self.progress("completed")

"""Shared configuration, random seeds, and SHA-256 provenance helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping without executing YAML object constructors.

    A missing file raises FileNotFoundError; malformed YAML raises YAMLError.
    Empty documents and non-mapping roots raise ValueError.
    """
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must contain a YAML mapping at its root")
    return cfg


def set_seeds(seed: int) -> None:
    """Seed Python and NumPy; pass the same seed to future model random_state.

    This seeds NumPy's legacy global RNG. Future default_rng instances must be
    constructed explicitly with the configured seed as well.
    """
    random.seed(seed)
    np.random.seed(seed)


def sha256_of_file(path: str | Path) -> str:
    """Hash raw file bytes in bounded chunks, including empty/binary files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash a sorted UTF-8 YAML dump, independent of mapping insertion order.

    The full configuration is hashed, including paths. Once preprocessing is
    implemented, include its version/settings in this mapping before hashing.
    Pinned PyYAML versions keep serialization consistent across environments.
    """
    serialized = yaml.safe_dump(
        cfg, sort_keys=True, allow_unicode=True, default_flow_style=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

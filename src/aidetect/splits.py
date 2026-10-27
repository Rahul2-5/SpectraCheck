"""Balanced, indivisible-group splitting and once-only split provenance."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from aidetect.io_utils import sha256_of_file


def balance_classes(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Retain every eligible AI image; reproducibly drop only excess real rows."""
    if not df["label"].isin([0, 1]).all() or df["id"].duplicated().any():
        raise ValueError("Balance requires unique IDs and binary labels")
    ordered = df.sort_values("id")
    real, ai = ordered[ordered["label"] == 0], ordered[ordered["label"] == 1]
    if ai.empty or len(real) < len(ai):
        raise ValueError("Need at least as many eligible real images as AI images; collect more real data")
    chosen = real.sample(n=len(ai), random_state=seed)
    return pd.concat([chosen, ai]).sort_values("id").reset_index(drop=True)


def make_group_splits(df: pd.DataFrame, proportions: dict[str, float], seed: int, *,
                      tolerance: float = 0.01, time_limit: float = 60) -> pd.DataFrame:
    """Allocate whole duplicate groups, with exact balance in EVERY split.

    Standard grouped-CV heuristics do not guarantee this combination of exact
    class balance and 70/15/15 tolerance. Use a small integer allocation over
    group count patterns (e.g. singleton real, singleton AI, mixed pair), then
    seed-shuffle groups within each pattern. Impossible constraints raise.
    """
    names = ["train", "val", "test"]
    if set(proportions) != set(names) or not np.isclose(sum(proportions.values()), 1):
        raise ValueError("Split proportions must contain train, val, test summing to one")
    p = np.asarray([proportions[name] for name in names], dtype=float)
    if (p <= 0).any() or tolerance < 0 or (p <= tolerance).any():
        raise ValueError("Invalid split proportions or tolerance")
    counts = df["label"].value_counts().to_dict()
    if set(counts) != {0, 1} or counts[0] != counts[1] or df["id"].duplicated().any() or df["group_id"].isna().any():
        raise ValueError("Grouped splitting requires balanced labels, unique IDs, and group IDs")
    groups = df.groupby(["group_id", "label"]).size().unstack(fill_value=0)
    patterns = defaultdict(list)
    for group, values in groups.iterrows():
        patterns[(int(values[0]), int(values[1]))].append(group)
    patterns = dict(sorted(patterns.items()))
    vectors = np.asarray(list(patterns), dtype=int)
    available = np.asarray([len(values) for values in patterns.values()])
    n_patterns, n_class = len(patterns), counts[0]
    # Three integer allocations per pattern, plus positive/negative deviations.
    n_integer = n_patterns * 3
    size = n_integer + 6
    objective = np.r_[np.zeros(n_integer), np.ones(6)]
    rows, lower, upper = [], [], []
    for index, total in enumerate(available):
        row = np.zeros(size)
        row[3 * index:3 * index + 3] = 1
        rows.append(row); lower.append(total); upper.append(total)
    for split in range(3):
        balance = np.zeros(size)
        balance[split:n_integer:3] = vectors[:, 0] - vectors[:, 1]
        rows.append(balance); lower.append(0); upper.append(0)
        count = np.zeros(size)
        count[split:n_integer:3] = vectors[:, 0]
        rows.append(count.copy())
        lower.append(max(1, np.ceil((p[split] - tolerance) * n_class - 1e-9)))
        upper.append(np.floor((p[split] + tolerance) * n_class + 1e-9))
        count[n_integer + 2 * split:n_integer + 2 * split + 2] = [-1, 1]
        rows.append(count); lower.append(p[split] * n_class); upper.append(p[split] * n_class)
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("Sample is too small for requested split proportions/tolerance")
    result = milp(objective, integrality=np.r_[np.ones(n_integer), np.zeros(6)],
        bounds=Bounds(np.zeros(size), np.r_[np.repeat(available, 3), np.full(6, np.inf)]),
        constraints=LinearConstraint(np.asarray(rows), lower, upper),
        options={"time_limit": time_limit})
    if not result.success:
        raise ValueError(f"Cannot split whole duplicate groups with exact balance/tolerance: {result.message}")
    allocation = np.rint(result.x[:n_integer]).astype(int).reshape(n_patterns, 3)
    rng, assignment = np.random.default_rng(seed), {}
    for index, values in enumerate(patterns.values()):
        shuffled = rng.permutation(sorted(values)).tolist()
        offset = 0
        for split, amount in zip(names, allocation[index]):
            for group in shuffled[offset:offset + amount]:
                assignment[group] = split
            offset += amount
    output = df[["id", "label", "group_id"]].copy()
    output["split"] = output["group_id"].map(assignment)
    output = output[["id", "label", "split", "group_id"]].sort_values("id").reset_index(drop=True)
    validate_splits(output, proportions, tolerance)
    return output


def validate_splits(df: pd.DataFrame, proportions: dict[str, float], tolerance: float) -> dict:
    """Assert group separation, exact per-split balance and proportion bounds."""
    if df.empty or df["id"].duplicated().any() or df["split"].isna().any() or df["group_id"].isna().any():
        raise ValueError("Invalid split IDs or missing assignments")
    if set(df["split"]) != set(proportions) or not df["label"].isin([0, 1]).all():
        raise ValueError("Invalid split names or labels")
    if (df.groupby("group_id")["split"].nunique() != 1).any():
        raise ValueError("A duplicate group crosses splits")
    table = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    for split, proportion in proportions.items():
        if set(table.columns) != {0, 1} or table.loc[split, 0] != table.loc[split, 1]:
            raise ValueError("A split is not exactly class-balanced")
        if abs(len(df[df["split"] == split]) / len(df) - proportion) > tolerance + 1e-9:
            raise ValueError("Split proportions exceed the configured tolerance")
    return {name: {"real": int(row[0]), "ai": int(row[1])} for name, row in table.iterrows()}


def freeze_splits(splits: pd.DataFrame, path: str | Path, hash_path: str | Path,
                  metadata: dict) -> str:
    """Create splits + hash + provenance once; reject partial or changed records."""
    path, hash_path = Path(path), Path(hash_path)
    metadata_path = hash_path.with_suffix(".json")
    payload = splits.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    record = {**metadata, "splits_sha256": digest}
    existing = [item.exists() for item in (path, hash_path, metadata_path)]
    if any(existing):
        if not all(existing):
            raise ValueError("Partial split freeze exists; inspect it before retrying")
        if (sha256_of_file(path) != digest or hash_path.read_text().strip() != digest
            or json.loads(metadata_path.read_text(encoding="utf-8")) != record):
            raise ValueError("Frozen splits or their provenance changed; downstream artifacts are invalid")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    with metadata_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with hash_path.open("x", encoding="utf-8") as stream:
        stream.write(digest + "\n")
    return digest

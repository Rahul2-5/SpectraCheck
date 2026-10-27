"""Group separation, exact balance, and immutable split records."""

import json

import pandas as pd
import pytest

from aidetect.splits import balance_classes, freeze_splits, make_group_splits, validate_splits

PROPORTIONS = {"train": .70, "val": .15, "test": .15}


@pytest.fixture
def balanced():
    frame = pd.DataFrame([{"id": f"{label}_{i:03}", "label": label, "group_id": f"{label}_{i:03}"}
                          for label in (0, 1) for i in range(100)])
    for label in (0, 1):
        frame.loc[(frame["label"] == label) & (frame["id"].str.endswith("001")), "group_id"] = f"{label}_000"
    return frame


def test_balance_drops_only_excess_real_and_is_order_independent(balanced):
    extra = pd.DataFrame([{"id": "extra", "label": 0, "group_id": "extra"}])
    frame = pd.concat([balanced, extra])
    result = balance_classes(frame, 42)
    assert result["label"].value_counts().to_dict() == {0: 100, 1: 100}
    assert set(result[result["label"] == 1]["id"]) == set(balanced[balanced["label"] == 1]["id"])
    pd.testing.assert_frame_equal(result, balance_classes(frame.sample(frac=1), 42))
    with pytest.raises(ValueError, match="collect more real"):
        balance_classes(balanced.drop(index=0), 42)


def test_group_split_exact_balance_tolerance_and_repeatability(balanced):
    result = make_group_splits(balanced, PROPORTIONS, 42)
    counts = validate_splits(result, PROPORTIONS, .01)
    assert all(row["real"] == row["ai"] for row in counts.values())
    assert (result.groupby("group_id")["split"].nunique() == 1).all()
    assert set(result["id"]) == set(balanced["id"])
    pd.testing.assert_frame_equal(result, make_group_splits(balanced.sample(frac=1), PROPORTIONS, 42))
    assert not result.equals(make_group_splits(balanced, PROPORTIONS, 43))


def test_mixed_label_groups_stay_together(balanced):
    balanced.loc[balanced["id"].isin(["0_050", "1_050"]), "group_id"] = "mixed"
    result = make_group_splits(balanced, PROPORTIONS, 42)
    assert result[result["group_id"] == "mixed"]["split"].nunique() == 1


def test_impossible_group_constraints_fail_clearly(balanced):
    balanced["group_id"] = balanced["label"].astype(str)
    with pytest.raises(ValueError, match="Cannot split whole duplicate groups"):
        make_group_splits(balanced, PROPORTIONS, 42)
    tiny = pd.DataFrame({"id": ["r", "a"], "label": [0, 1], "group_id": ["r", "a"]})
    with pytest.raises(ValueError, match="too small"):
        make_group_splits(tiny, PROPORTIONS, 42)


def test_frozen_split_rerun_unchanged_and_tamper_rejected(tmp_path, balanced):
    splits = make_group_splits(balanced, PROPORTIONS, 42)
    path, lock = tmp_path / "splits.csv", tmp_path / "artifacts/splits.sha256"
    metadata = {"config_hash": "config", "active_manifest_sha256": "manifest"}
    digest = freeze_splits(splits, path, lock, metadata)
    timestamps = [p.stat().st_mtime_ns for p in (path, lock, lock.with_suffix(".json"))]
    assert freeze_splits(splits, path, lock, metadata) == digest
    assert timestamps == [p.stat().st_mtime_ns for p in (path, lock, lock.with_suffix(".json"))]
    assert json.loads(lock.with_suffix(".json").read_text())["splits_sha256"] == digest
    path.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        freeze_splits(splits, path, lock, metadata)


def test_partial_freeze_and_group_crossing_are_rejected(tmp_path, balanced):
    splits = make_group_splits(balanced, PROPORTIONS, 42)
    path = tmp_path / "splits.csv"
    path.write_text("partial")
    with pytest.raises(ValueError, match="Partial"):
        freeze_splits(splits, path, tmp_path / "splits.sha256", {})
    duplicates = splits[splits["group_id"] == "0_000"].index
    splits.loc[duplicates[0], "split"] = "test"
    splits.loc[duplicates[1], "split"] = "train"
    with pytest.raises(ValueError, match="crosses"):
        validate_splits(splits, PROPORTIONS, .01)
